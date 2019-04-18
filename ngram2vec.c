//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

#define MAX_STRING 30
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
#define MAX_NGRAM 10

const int ngram_hash_size = 60000000;  // Maximum 60 * 0.7 = 42M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct ngram_info {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct ngram_info *ngram_infos;
int ngrams;
int binary = 0, debug_mode = 2, window = 5, num_threads = 12, min_reduce = 1;
int *ngram_hash;
long long ngram_max_size = 1000, vocab_size = 0, layer1_size = 100, ngram_size;
long long total_words = 0, word_count_actual = 0, iter = 5, file_size = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
const int neg_table_size = 1e8;
int *neg_table;

void InitUnigramTable() {
  int a, i;
  double total_words_pow = 0;
  double d1, power = 0.75;
  neg_table = (int *)malloc(neg_table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) total_words_pow += pow(ngram_infos[a].cn, power);
  i = 0;
  d1 = pow(ngram_infos[i].cn, power) / total_words_pow;
  for (a = 0; a < neg_table_size; a++) {
    neg_table[a] = i;
    if (a / (double)neg_table_size > d1) {
      i++;
      d1 += pow(ngram_infos[i].cn, power) / total_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % ngram_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (ngram_hash[hash] == -1) return -1;
    if (!strcmp(word, ngram_infos[ngram_hash[hash]].word)) return ngram_hash[hash];
    hash = (hash + 1) % ngram_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING*MAX_NGRAM + MAX_NGRAM) length = MAX_STRING*MAX_NGRAM + MAX_NGRAM;
  ngram_infos[ngram_size].word = (char *)calloc(length, sizeof(char));
  strcpy(ngram_infos[ngram_size].word, word);
  ngram_infos[ngram_size].cn = 0;
  ngram_size++;
  // Reallocate memory if needed
  if (ngram_size + 2 >= ngram_max_size) {
    ngram_max_size += 1000;
    ngram_infos = (struct ngram_info *)realloc(ngram_infos, ngram_max_size * sizeof(struct ngram_info));
  }
  hash = GetWordHash(word);
  while (ngram_hash[hash] != -1) hash = (hash + 1) % ngram_hash_size;
  ngram_hash[hash] = ngram_size - 1;
  return ngram_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    int a_has_space = (strchr(((struct ngram_info *)a)->word, ' ') != NULL);
    int b_has_space = (strchr(((struct ngram_info *)b)->word, ' ') != NULL);
    if (a_has_space < b_has_space) {
      return -1;
    }
    if (a_has_space > b_has_space) {
      return 1;
    }
    return ((struct ngram_info *)b)->cn - ((struct ngram_info *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortNgram() {
  int a;
  unsigned int hash;
  char is_multi_gram = 0;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&ngram_infos[1], ngram_size - 1, sizeof(struct ngram_info), VocabCompare);
  for (a = 0; a < ngram_hash_size; a++) ngram_hash[a] = -1;
  vocab_size = 0;
  for (a = 0; a < ngram_size; a++) {
    // Hash will be re-computed, as after the sorting it is not actual
    if ((!is_multi_gram) && (strchr(ngram_infos[a].word, ' ') == NULL)) {
      vocab_size++;
    } else {
      is_multi_gram = 1;
    }
    hash=GetWordHash(ngram_infos[a].word);
    while (ngram_hash[hash] != -1) hash = (hash + 1) % ngram_hash_size;
    ngram_hash[hash] = a;
  }
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    ngram_infos[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    ngram_infos[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceNgram() {
  min_reduce *= 2;
  int a, b = 1;
  unsigned int hash;
  for (a = 1; a < ngram_size; a++) if (ngram_infos[a].cn > 1) {
    ngram_infos[b].cn = ngram_infos[a].cn/2;
    ngram_infos[b].word = ngram_infos[a].word;
    b++;
  } else {
    if (rand()%2) {
      free(ngram_infos[a].word);
    } else {
      ngram_infos[b].cn = ngram_infos[a].cn;
      ngram_infos[b].word = ngram_infos[a].word;
      b++;
    }
  }
  ngram_size = b;
  for (a = 0; a < ngram_hash_size; a++) ngram_hash[a] = -1;
  for (a = 0; a < ngram_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(ngram_infos[a].word);
    while (ngram_hash[hash] != -1) hash = (hash + 1) % ngram_hash_size;
    ngram_hash[hash] = a;
  }
  fflush(stdout);
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = ngram_infos[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    ngram_infos[a].codelen = i;
    ngram_infos[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      ngram_infos[a].code[i - b - 1] = code[b];
      ngram_infos[a].point[i - b] = point[b] - vocab_size;//path from root to leaf
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void LearnVocabFromTrainFile() {
  srand(time(NULL));
  char word[MAX_STRING];
  char words[MAX_NGRAM][MAX_STRING];
  char ngram[MAX_STRING*MAX_NGRAM + MAX_NGRAM];
  FILE *fin;
  long long a, i;
  int n;
  int is_beginning_of_sentence = 1;
  for (a = 0; a < ngram_hash_size; a++) ngram_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  ngram_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    if (!strcmp(word, "</s>")) {
      is_beginning_of_sentence = 1;
      continue;
    }
    total_words++;
    if ((debug_mode > 1) && (total_words % 100000 == 0)) {
      printf("%lldK%c", total_words / 1000, 13);
      fflush(stdout);
    }
    if (is_beginning_of_sentence) {
      for (i = 0; i < ngrams; ++i) {
        strcpy(words[i], "");
      }
      is_beginning_of_sentence = 0;
    } else {
      for ( i = 0; i+1 < ngrams; ++i ) {
        strcpy(words[i], words[i+1]);
      }
    }
    strcpy(words[ngrams-1] , word);
    for (n = 1; n <= ngrams; ++n) {
      if (min_reduce > 1 && rand()%min_reduce) {
        continue;
      }
      strcpy(ngram, "");
      for (i = ngrams - n; i < ngrams; ++i) {
        if (words[i] == 0) {
	  ngram[0] = 0;
	  break;
	}
        if (ngram[0]) {
	  strcat(ngram, " ");
	}
	strcat(ngram, words[i]);
      }
      if (ngram[0] == 0) {
        continue;
      }
      i = SearchVocab(ngram);
      if (i == -1) {
        //AddWordToVocab(ngram);
        a = AddWordToVocab(ngram);
        ngram_infos[a].cn = 1;
      } else ngram_infos[i].cn++;
      while (ngram_size > ngram_hash_size * 0.7) ReduceNgram();
    }
//    if (!strcmp(word, "</s>")) {
//      is_beginning_of_sentence = 1;
//    } else {
//      is_beginning_of_sentence = 0;
//    }
  }
  SortNgram();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("ngram size: %lld\n", ngram_size);
    printf("Words in train file: %lld\n", total_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", ngram_infos[i].word, ngram_infos[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < ngram_hash_size; a++) ngram_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &ngram_infos[a].cn, &c);
    i++;
  }
  SortNgram();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", total_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  a = posix_memalign((void **)&syn0, 128, (long long)ngram_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  if (hs) {
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1[a * layer1_size + b] = 0;
  }
  if (negative>0) {
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1neg[a * layer1_size + b] = 0;
  }
  for (a = 0; a < ngram_size; a++) for (b = 0; b < layer1_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
  }
  CreateBinaryTree();
}

void *TrainModelThread(void *id) {
  long long a, b, d, cw, wid, last_word, sentence_length = 0, sentence_position = 0;
  int y;
  long long word_count = 0, last_word_count = 0, sen[MAX_NGRAM][MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long)id;
  real f, g;
  clock_t now;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  char word[MAX_STRING];
  char words[MAX_NGRAM][MAX_STRING];
  char ngram[MAX_STRING*MAX_NGRAM + MAX_NGRAM];
  int is_beginning_of_sentence;
  int i, n;
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         word_count_actual / (real)(iter * total_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * total_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      is_beginning_of_sentence = 1;
      while (1) {
        ReadWord(word, fi);
        if (feof(fi)) break;
        if (!strcmp(word, "</s>")) {
	  is_beginning_of_sentence = 1;
	  break;
        } 
	++word_count;
        if (is_beginning_of_sentence) {
          for (i = 0; i < ngrams; ++i) {
            strcpy(words[i], "");
          }
	  is_beginning_of_sentence = 0;
        } else {
          for ( i = 0; i+1 < ngrams; ++i ) {
            strcpy(words[i], words[i+1]);
          }
        }
        strcpy(words[ngrams-1] , word);
        for (n = 1; n <= ngrams; ++n) {
          strcpy(ngram, "");
          for (i = ngrams - n; i < ngrams; ++i) {
            if (words[i] == 0) {
              ngram[0] = 0;
              break;
            }
            if (ngram[0]) {
              strcat(ngram, " ");
            }
            strcat(ngram, words[i]);
          }
          if (ngram[0] == 0) {
	    sen[n-1][sentence_length] = -1;
            continue;
          }
          wid = SearchVocab(ngram);
          if (wid != -1) {
            if (sample > 0) {
              real ran = (sqrt(ngram_infos[wid].cn / (sample * total_words)) + 1) * (sample * total_words) / ngram_infos[wid].cn;
              next_random = next_random * (unsigned long long)25214903917 + 11;
              if (ran < (next_random & 0xFFFF) / (real)65536) {
	        sen[n-1][sentence_length] = -1;
	      } else {
	        sen[n-1][sentence_length] = wid;
	      }
            } else {
	      sen[n-1][sentence_length] = wid;
	    }
          } 
        }
      
        // The subsampling randomly discards frequent words while keeping the ranking same
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }

    if (feof(fi) || (word_count > total_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }
    y = sen[0][sentence_position];
    if (y == -1) {
      sentence_position++;
      if (sentence_position >= sentence_length) {
        sentence_length = 0;
      }
      continue;
    }
      
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;
    for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
      c = sentence_position - window + a;
      if (c < 0) continue;
      if (c >= sentence_length) continue;
      for (n = 0; n < ngrams; ++n) { 
        // [c-n, c]
        if (c-n <= sentence_position && c >= sentence_position) {
          continue;
        }
        last_word = sen[n][c];
        if (last_word == -1) continue;
        l1 = last_word * layer1_size;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        // HIERARCHICAL SOFTMAX
        if (hs) for (d = 0; d < ngram_infos[y].codelen; d++) {
          f = 0;
          l2 = ngram_infos[y].point[d] * layer1_size;
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - ngram_infos[y].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = y;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = neg_table[(next_random >> 16) % neg_table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == y) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
        }
        // Learn weights input -> hidden
        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
      }
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

void TrainModel() {
  long a, b, c, d;
  FILE *fo;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  if (save_vocab_file[0] != 0) SaveVocab();
  if (output_file[0] == 0) return;
  InitNet();
  if (negative > 0) InitUnigramTable();
  start = clock();
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  fo = fopen(output_file, "wb");
  // Save the word vectors
  fprintf(fo, "%lld\t%lld\n", ngram_size, layer1_size);
  for (a = 0; a < ngram_size; a++) {
    fprintf(fo, "%s", ngram_infos[a].word);
    if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
    else for (b = 0; b < layer1_size; b++) fprintf(fo, "\t%lf", syn0[a * layer1_size + b]);
    fprintf(fo, "\n");
  }
  fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-ngram_infos <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-ngram_infos <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-ngrams <int>\n");
    printf("\t\tmax ngram (default = 1)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -iter 3\n\n");
    return 0;
  }
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  ngrams = 1;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-ngram_infos", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-ngram_infos", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-ngrams", argc, argv)) > 0) ngrams = atoi(argv[i + 1]);
  ngram_infos = (struct ngram_info *)calloc(ngram_max_size, sizeof(struct ngram_info));
  ngram_hash = (int *)calloc(ngram_hash_size, sizeof(int));
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() neg_table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  return 0;
}
