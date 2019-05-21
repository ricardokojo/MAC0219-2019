#include<stdio.h>
#include<iostream>
#include<stdlib.h>
#include<complex>
#include<png.h>
#define ITERATIONS 1000


inline void setRGB(png_byte *ptr, float val)
{
	int v = (int)(val * 767);
	if (v < 0) v = 0;
	if (v > 767) v = 767;
	int offset = v % 256;

	if (v<256) {
		ptr[0] = 0; ptr[1] = 0; ptr[2] = offset;
	}
	else if (v<512) {
		ptr[0] = 0; ptr[1] = offset; ptr[2] = 255-offset;
	}
	else {
		ptr[0] = offset; ptr[1] = 255-offset; ptr[2] = 0;
	}
}

int main(int argc, char *argv[])
{
	int ERROR=-100;
    if (argc < 9)
    {
		std::cout << "mbrot <C0_REAL> <C0_IMAG> <C1_REAL> <C1_IMAG> <W> <H> <CPU/GPU> <THREADS> <SAIDA>\n";
        return 1;
    }

    //Declaram-se e instanciam-se as variáveis de entrada:
    float C0_REAL, C0_IMAG, C1_REAL, C1_IMAG;
	int WIDTH, HEIGHT, THREADS;
    char *CPU_GPU, *SAIDA;

    C0_REAL = float(atof(argv[1]));
	C0_IMAG = float(atof(argv[2]));
	C1_REAL = float(atof(argv[3]));
	C1_IMAG = float(atof(argv[4]));
	WIDTH = atoi(argv[5]);
	HEIGHT = atoi(argv[6]);
	CPU_GPU = argv[7];
	THREADS = atoi(argv[8]);
    SAIDA = argv[9];
	
	std::cout << C0_REAL << C0_IMAG << C1_REAL << C1_IMAG << WIDTH << HEIGHT << CPU_GPU << THREADS << SAIDA;
	std::complex<float> current = 0;
	std::complex<float> last = 0;
	std::complex<float> c = 0;
	float d_x = (C1_REAL - C0_REAL) / WIDTH;
	float d_y = (C1_IMAG - C0_IMAG) / HEIGHT;
	bool mandel = 1;
	
	FILE *fp = fopen("asd.png", "wb"); 
	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL , NULL , NULL);
	if(!png_ptr) return(ERROR);
	png_infop info_ptr = png_create_info_struct(png_ptr);
	if(!info_ptr)
	{
		png_destroy_write_struct(&png_ptr,(png_infopp)NULL);
		return(ERROR);
	}

	float *buffer = (float *) malloc(WIDTH * HEIGHT * sizeof(float));
	if (buffer == NULL) {
		fprintf(stderr, "Could not create image buffer\n");
		return -1;
	}

	png_init_io(png_ptr, fp);

	info_ptr = png_create_info_struct(png_ptr);

	png_set_IHDR(png_ptr, info_ptr, WIDTH, HEIGHT,
			8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
			PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

	png_write_info(png_ptr, info_ptr);

	

	for (int y = 0; y < HEIGHT; ++y) {
		for (int x = 0; x < WIDTH; ++x) {
			mandel = 1;
			c = (C0_REAL + (x * d_x), C0_IMAG + (y * d_y)); 
			for (int t = 1; t < ITERATIONS; ++t) {
				current = last * last + c;
				if (std::abs(current) > 2) {
					mandel = 0;

					break; // pintar baseado no t em que parou
				}
				buffer[y* WIDTH + x]= t/(float) ITERATIONS;
				std::cout << t/ (float) ITERATIONS << "\n";
				last = current;
			}

			if (mandel) {
				buffer[y * WIDTH + x]=0;
			}
		}
	}

	png_bytep row = (png_bytep) malloc(3 * WIDTH * sizeof(png_byte));

	// Write image data
	int x;
	int y;
	for (y=0 ; y<HEIGHT ; y++) {
		for (x=0 ; x<WIDTH ; x++) {
			setRGB(&(row[x*3]), buffer[y*WIDTH + x]);
		}
		png_write_row(png_ptr, row);
	}

	// End write
	png_write_end(png_ptr, NULL);
	return 0;
}
