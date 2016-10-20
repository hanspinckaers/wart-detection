// build with clang -Ofast -o kmajority kmajority.c
// if no clang is available try gcc
// debug clang -Ofast -o kmajority kmajority.c (something with -g)
// lldb ./kmajority
// process launch -- <args>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int CountOnesFromInteger(unsigned int value) {
    int count;
    for (count = 0; value != 0; count++, value &= value-1);
    return count;
}

unsigned int HammingDistance(int *a, int *b, int vec_size) {
	int dist = 0;
	for (int i = 0; i < vec_size; ++i){
		dist += CountOnesFromInteger(a[i] ^ b[i]);
	}
	return dist;
}

void argToArray(int **vectors, char *arg, int vec_size, int vec_count) {
	char *p = strtok(arg, "\n");
	char *numbers[vec_count];
	for (int i = 0; i < vec_count; i++)
		numbers[i] = (char *)malloc(vec_size * sizeof(char) * 4);

	int vec_i = 0;
	
	while (p != NULL) {
		numbers[vec_i++] = p;
		p = strtok(NULL, "\n");
	}
	
	for (int i = 0; i < vec_count; ++i) {
		char *n = strtok(numbers[i], " ");
		int n_i = 0;
		while (n != NULL) {
			int number = strtol(n, NULL, 10);
			vectors[i][n_i++] = number;
			n = strtok(NULL, " ");
		}	
	}
}

int main( int argc, char* argv[] )
{
	if (argc > 3) {
		int vec_size = atoi(argv[1]);
		int vec_count = atoi(argv[2]);
		int cen_count = atoi(argv[3]);

		int* vectors[vec_count];
		int* centroids[cen_count];

		for (int i = 0; i < vec_count; i++)
        	vectors[i] = (int *)malloc(vec_size * sizeof(int));

		for (int i = 0; i < cen_count; i++)
        	centroids[i] = (int *)malloc(vec_size * sizeof(int));

		// loading whole file in memory is wasteful, better would be to combine file read with argToArray function
		FILE *f = fopen(argv[4], "rb");
		fseek(f, 0, SEEK_END);
		long fsize = ftell(f);
		fseek(f, 0, SEEK_SET);  //same as rewind(f);
		char *vec_string = (char *)malloc(sizeof(char) * (fsize + 2));
		fread(vec_string, fsize, 1, f);
		fclose(f);

		f = fopen(argv[5], "rb");
		fseek(f, 0, SEEK_END);
		fsize = ftell(f);
		fseek(f, 0, SEEK_SET);  //same as rewind(f);
		char *cen_string = (char *)malloc(sizeof(char) * (fsize + 1));
		fread(cen_string, fsize, 1, f);
		fclose(f);

		argToArray(vectors, vec_string, vec_size, vec_count);	
		argToArray(centroids, cen_string, vec_size, cen_count);

		free(vec_string);
		free(cen_string);

		int v_i = 0;
		for (v_i = 0; v_i < vec_count; ++v_i) {
			if (v_i > 0) {
				printf(" ");
			}
			int *vec = vectors[v_i];
			int smallest_dist = 0;
			int closest_cen = 0;
			for (int c_i = 0; c_i < cen_count; ++c_i) {
				int *cen = centroids[c_i];
				int dist = HammingDistance(vec, cen, vec_size);
				if (c_i == 0) 
					smallest_dist = dist;
				else if (dist < smallest_dist) {
					closest_cen = c_i;
					smallest_dist = dist;
				}
			}
			printf("%i", closest_cen);
		}
	} 
	else {
		printf("No arguments passed\n");
		return 1;
	}
	return 0;
}


