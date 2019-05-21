#include<stdio.h>
#include<iostream>
#include<stdlib.h>
#include<complex>
#include<png.h>
#include<string>

using namespace std;

#define ITERATIONS 1000

inline void setColorValue(png_byte *ptr, float val){
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

int printImage(string file_name, int w, int h, float* buffer_image){
	
	FILE *file = NULL;
	png_structp image_ptr = NULL;
	png_infop info_ptr = NULL;
	png_bytep buffer_row = NULL;
	
	int status=1;
	file = fopen(file_name.c_str(), "wb");
	if (file == NULL) {
		 cerr << "Falha arquivo destinado para a escrita da imagem: " << file_name << endl;
		 status=0;
	}

	if(status){
		image_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
		if (image_ptr == NULL) {
			cerr << "Falha ao alocar uma estrutura de escrita." << endl;
			status=0;}
	}

	if(status){
		info_ptr = png_create_info_struct(image_ptr);
		if (info_ptr == NULL) {
			cerr << "Falha ao alocar uma estrutura de meta-dados para a imagem." << endl;
			status=0;}
	}

	if(status){
		if (setjmp(png_jmpbuf(image_ptr))) {
			cerr << "Erro durante a criação da imagem." << endl;
			status=0;}
	}

	if(status){
		png_init_io(image_ptr, file);
		png_set_IHDR(image_ptr, info_ptr, w, h,
			8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
			PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
		png_write_info(image_ptr, info_ptr);

		buffer_row = (png_bytep) malloc(3 * w * sizeof(png_byte));
		int x, y;
		for (y=0 ; y<h ; y++) {
			for (x=0 ; x<w ; x++) {
				setColorValue(&(buffer_row[x*3]), buffer_image[y*w + x]);}
			png_write_row(image_ptr, buffer_row);
		}
		png_write_end(image_ptr, NULL);
	}

	if (file != NULL) fclose(file);
	if (info_ptr != NULL) png_free_data(image_ptr, info_ptr, PNG_FREE_ALL, -1);
	if (image_ptr != NULL) png_destroy_write_struct(&image_ptr, (png_infopp)NULL);
	if (buffer_row != NULL) free(buffer_row);


return status-1;
}

float* mbrot_func(float c0_r, float c0_i, float c1_r, float c1_i, int w, int h, int iteractions){
	float *buffer_image = (float *) malloc(w * h * sizeof(float));
	if (buffer_image == NULL) {
		cerr << "Falha ao criar o Buffer da imagem." << endl;
		return NULL;}
	complex<float> current=0;
	complex<float> last = 0;
	complex<float> c = 0;
	float d_x = (c1_r - c0_r) / (float) w;
	float d_y = (c1_i - c0_i) / (float) h;
	bool mandel = 1;
	int max_t=0;
	cout <<"w "<< w << endl;
	cout << "h " << h << endl;
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			mandel = 1;
			cout << "y_comp" << c0_i + (y * d_y) << endl;
			cout << "x_comp" << c0_r + (x * d_x) << endl;

			c = (c0_r + (x * d_x), c0_i + (y * d_y));
			cout << "c"<< c << endl;
			last=0;
			for (int t = 1; t < iteractions; ++t) {
				current = last * last + c;
				if (abs(current) > 2) {
					mandel = 0;
					if(t>max_t){max_t=t;}
					buffer_image[y*w + x]= (float) t;
					break; // pintar baseado no t em que parou
				}
				last = current;
			}
			if(mandel){
				buffer_image[y*w + x]=0.0;
			}
		}
	}

	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			buffer_image[y*w + x]=buffer_image[y*w + x]/ (float) max_t;
		}
	}

	return buffer_image;
}

int main(int argc, char *argv[])
{
	// float *buffer = (float *) malloc(3 * 3 * sizeof(float));
	// buffer[0]=0;
	// buffer[1]=0.1;
	// buffer[2]=0.2;
	// buffer[3]=0.3;
	// buffer[4]=0.4;
	// buffer[5]=0.5;
	// buffer[6]=0.6;
	// buffer[7]=0.7;
	// buffer[8]=0.8;
	float* buffer=mbrot_func( 0.011,110,-0.802, -0.177, 1000,1000,1000);
	string file_name="testinho.png";
	cout << buffer[100000] << endl;
	return printImage(file_name,1000,1000, buffer);
}



