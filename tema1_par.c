// Author: APD team, except where source was noted

#include "helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>

#define CONTOUR_CONFIG_COUNT    16
#define FILENAME_MAX_SIZE       50
#define STEP                    8
#define SIGMA                   200
#define RESCALE_X                   2048
#define RESCALE_Y               2048

#define CLAMP(v, min, max) if(v < min) { v = min; } else if(v > max) { v = max; }

// structura creata pentru a o trimite thread ului cu toate informatiile necesare,
// pentru a nu le pune ca si variabile globale
typedef struct {
    int N;
    ppm_image *scaled_image;
    ppm_image *image;
    int ok;
    unsigned char **grid;
    int p, q;
    ppm_image **contour_map;
    pthread_barrier_t *b;

} ThreadData;

// o structura ce o inglobeaza si pe prima si are si id ul
typedef struct {
    ThreadData *data;
    int thread_id;
} ThreadArgs;


// Creates a map between the binary configuration (e.g. 0110_2) and the corresponding pixels
// that need to be set on the output image. An array is used for this map since the keys are
// binary numbers in 0-15. Contour images are located in the './contours' directory.
ppm_image **init_contour_map() {
    ppm_image **map = (ppm_image **)malloc(CONTOUR_CONFIG_COUNT * sizeof(ppm_image *));
    if (!map) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
        char filename[FILENAME_MAX_SIZE];
        sprintf(filename, "./contours/%d.ppm", i);
        map[i] = read_ppm(filename);
    }

    return map;
}

// Updates a particular section of an image with the corresponding contour pixels.
// Used to create the complete contour image.
void update_image(ppm_image *image, ppm_image *contour, int x, int y) {
    for (int i = 0; i < contour->x; i++) {
        for (int j = 0; j < contour->y; j++) {
            int contour_pixel_index = contour->x * i + j;
            int image_pixel_index = (x + i) * image->y + y + j;

            image->data[image_pixel_index].red = contour->data[contour_pixel_index].red;
            image->data[image_pixel_index].green = contour->data[contour_pixel_index].green;
            image->data[image_pixel_index].blue = contour->data[contour_pixel_index].blue;
        }
    }
}

// Calls `free` method on the utilized resources.
void free_resources(ppm_image *image, ppm_image **contour_map, unsigned char **grid, int step_x) {
    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
        free(contour_map[i]->data);
        free(contour_map[i]);
    }
    free(contour_map);

    for (int i = 0; i <= image->x / step_x; i++) {
        free(grid[i]);
    }
    free(grid);

    free(image->data);
    free(image);
}

// aici am definit functia thread urilor, pe care fiecare o foloseste
// am pararelizat 3 functii: rescale, sample_grid, march
void *thread_function(void *arg) {
    // structura mare primita ca si parametru
    ThreadArgs *args = (ThreadArgs *)arg;
    ThreadData *data = args->data;
    // am extras datele
    int thread_id = args->thread_id;
    pthread_barrier_t *b = data->b;
    // inclusiv bariera

    // ok ul este folosit pentru a face rescale doar atunci cand imaginea era mai mare
    if (data->ok == 1) { 
        int pr = data->scaled_image->x;
        int start = thread_id * (pr / data->N);
        int end = (thread_id == data->N - 1) ? pr : (thread_id + 1) * (pr / data->N); // ajusteaza pentru ultimul thread
        // impart portiuni de calcul fiecarui thread din matrice
        uint8_t sample[3];
        // se efectueaza codul similar doar ca pe portiuni si alte variabile din structura
        for (int i = start; i < end; i++){
            for (int j = 0; j < data->scaled_image->y; j++) {
                float u = (float)i / (float)(data->scaled_image->x - 1);
                float v = (float)j / (float)(data->scaled_image->y - 1);
                sample_bicubic(data->image, u, v, sample);

                data->scaled_image->data[i * data->scaled_image->y + j].red = sample[0];
                data->scaled_image->data[i * data->scaled_image->y + j].green = sample[1];
                data->scaled_image->data[i * data->scaled_image->y + j].blue = sample[2];                
            }
        }
        // Adaug o bariera pentru a ma asigura ca toate thread-urile au terminat procesarea inainte de a continua
        // putand sa folosesc complet datale prelucrate si neexistand sa iau o valoare veche
        pthread_barrier_wait(b);
    }
    // sample_grid
    // codul este similar doar ca redus la un singur for mare pe portiunea fiecarui thread
    int start_i = thread_id * (data->p / data->N);
    int end_i = (thread_id == data->N - 1) ? data->p : (thread_id + 1) * (data->p / data->N); // Ajustez pentru ultimul thread

    for (int i = start_i; i < end_i; i++) {
        for (int j = 0; j < data->q; j++) {
            ppm_pixel curr_pixel = data->scaled_image->data[i * STEP * data->scaled_image->y + j * STEP];

            unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;
            if (curr_color > SIGMA) {
                data->grid[i][j] = 0;
            } else {
                data->grid[i][j] = 1;
            }
        }

        ppm_pixel curr_pixel = data->scaled_image->data[i * STEP * data->scaled_image->y + data->scaled_image->x - 1];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > SIGMA) {
            data->grid[i][data->q] = 0;
        } else {
            data->grid[i][data->q] = 1;
        }

        curr_pixel = data->scaled_image->data[(data->scaled_image->x - 1) * data->scaled_image->y + i * STEP];

        curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > SIGMA) {
            data->grid[data->p][i] = 0;
        } else {
            data->grid[data->p][i] = 1;
        }
    }
    // Din nou, o alta bariera pentru a astepta toate threadurile ca au termina
    pthread_barrier_wait(b);

    // march
    // pararelizata pe portiuni

    for (int i = start_i; i < end_i; i++) {
        for (int j = 0; j < data->q; j++) {
            unsigned char k = 8 * data->grid[i][j] + 4 * data->grid[i][j + 1] + 2 * data->grid[i + 1][j + 1] + 1 * data->grid[i + 1][j];
            update_image(data->scaled_image, data->contour_map[k], i * STEP, j * STEP);
        }
    }


    return NULL;
}


int main(int argc, char *argv[]) {
    ThreadData data;
    data.ok = 1;
    data.b = (pthread_barrier_t *)malloc(sizeof(pthread_barrier_t));
    // am alocat bariera
    
    if (argc < 4) {
        fprintf(stderr, "Usage: ./tema1 <in_file> <out_file> <P>\n");
        return 1;
    }
    // am extras numarul de threaduri primit ca si parametru de exercitiu
    data.N = atoi(argv[3]);
    // initializiera vectorului de threaduri si a barierei
	pthread_t tid[data.N];
    pthread_barrier_init(data.b, NULL, data.N);

    // se creeaza vectorul de id uri
	int *thread_id = (int *)malloc(data.N * sizeof(int));
    
    // functie deja existenta doar ca le trec direct in structura
    data.image = read_ppm(argv[1]);

    // 0. Initialize contour map
    data.contour_map = init_contour_map();

    // 1. Rescale the image

    // urmatorul cod este preluat 1 la 1 cu cel existent in rescale_image pentru a initializa scaled_image sau
    // de a aloca si atribui datele necesar, etapa efectuata o singura data, neputand fi trecuta in functia threadului
    // doar daca o particularizam pentru un singur thread...
    uint8_t sample[3];

    if (data.image->x <= RESCALE_X && data.image->y <= RESCALE_Y){
        data.scaled_image = data.image;
        data.ok = 0;
    } else {
        data.scaled_image = (ppm_image *)malloc(sizeof(ppm_image));
        if (!data.scaled_image){
            fprintf(stderr, "Unable to allocate memory\n");
            exit(1);
        }
        data.scaled_image->x = RESCALE_X;
        data.scaled_image->y = RESCALE_Y;

        data.scaled_image->data = (ppm_pixel*)malloc(data.scaled_image->x * data.scaled_image->y * sizeof(ppm_pixel)); 
        if (!data.scaled_image) {
            fprintf(stderr, "Unable to allocate memory\n");
            exit(1);
        }
    }
    
    // 2. Sample the grid
    // aici la fel, partea de inceput din sample_grid, in care e initializat si alocat
    data.p = data.scaled_image->x / STEP;
    data.q = data.scaled_image->y / STEP;

    data.grid = (unsigned char **)malloc((data.p+1) * sizeof(unsigned char*));
    if (!data.grid) {
        fprintf(stderr, "Unbale to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i <= data.p; i++) {
        data.grid[i] = (unsigned char *)malloc((data.q + 1) * sizeof(unsigned char));
        if (!data.grid[i]) {
            fprintf(stderr, "Unable to allocate memory\n");
            exit(1);
        }
    }


    // crearea threadurilor impreuna cu structura ThreadArgs
    ThreadArgs args[data.N];

    for(int i = 0; i < data.N; i++) {
        args[i].data = &data;
        args[i].thread_id = i;
        pthread_create(&tid[i], NULL, thread_function, &args[i]);
    }
    // pornirea threadurilor
    for (int i = 0; i < data.N; i++) {
		pthread_join(tid[i], NULL);
	}


    // 4. Write output
    write_ppm(data.scaled_image, argv[2]);
    free(thread_id);
    free_resources(data.scaled_image, data.contour_map, data.grid, STEP);
	pthread_barrier_destroy(data.b);

    return 0;
}
