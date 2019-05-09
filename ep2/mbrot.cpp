#include<stdio.h>
#include<iostream>
#include<stdlib.h>
#include<complex>
#include<png.h>
#define ITERATIONS 1000

int main(int argc, char *argv[])
{
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
	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, (png_voidp)user_error_ptr,user_error_fn, user_warning_fn);
	if(!png_ptr) return(ERROR);
	png_infop info_ptr = png_create_info_struct(png_ptr);
	if(!info_ptr)
	{
		png_destroy_write_struct(&png_ptr,(png_infopp)NULL);
		return(ERROR);
	}
	png_init_io(png_prt, fp);

	for (int x = 0; x < WIDTH; ++x) {
		for (int y = 0; y < HEIGHT; ++y) {
			mandel = 1;
			c = (C0_REAL + (x * d_x), C0_IMAG + (y * d_y)); 
			for (int t = 1; t < ITERATIONS; ++t) {
				current = last * last + c;
				if (std::abs(current) > 2) {
					mandel = 0;
					break; // pintar baseado no t em que parou
				}
				last = current;
			}

			if (mandel) {
				// pintar de preto
			} else {
				// pintar de baseado no t em que parou 
			}
		}
	}
	return 0;
}
