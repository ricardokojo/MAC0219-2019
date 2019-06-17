//********************************************//
// MAC0219/5742 - EP3                         //
// EP3 - Mandelbrot                           //
// Bruna Bazaluk, Felipe Serras, Ricardo Kojo //
//********************************************//
//*Arquivo que contem as funções para adequação e savamento das imagens e outras funções auxiliares
//que são usada em ambos os conjuntos de demais funções (GPU e CPU) *//

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <png.h> // Manipulação de pngs
using namespace std;

// Função que transforma um vetor de valores entre 0 a 1 em cores na escala RGP
// Baseado na função apresentada em 	http:;;www.labbookpages.co.uk/software/imgProc/libPNG.html
inline void setColorValue(png_byte *ptr, double val)
{
	int v = (int)(val * 767);
	if (v < 0)
		v = 0;
	if (v > 767)
		v = 767;
	int offset = v % 256;

	if (v < 100)
	{
		ptr[0] = 0;
		ptr[1] = 0;
		ptr[2] = offset;
	}
	else if (v < 300)
	{
		ptr[0] = 0;
		ptr[1] = offset;
		ptr[2] = 255 - offset;
	}
	else
	{
		ptr[0] = offset;
		ptr[1] = 255 - offset;
		ptr[2] = 0;
	}
}

// Função que recebe uma um vetor de floats representado a imagem de buffer e o salva no arquivo png
// com o nome filname
int printImage(string file_name, int w, int h, float *buffer_image)
{
	// A cada oassi atualiza-se o inteiro status que controla se o fluxo de salvamento da imagem deve prosseguir.
	// Se houver algum erro num passo intermediário o fluxo é interrompido.

	FILE *file = NULL;
	png_structp image_ptr = NULL;
	png_infop info_ptr = NULL;
	png_bytep buffer_row = NULL;

	// Criação do Arquivo:
	int status = 1;
	file = fopen(file_name.c_str(), "wb");
	if (file == NULL)
	{
		cerr << "Falha arquivo destinado para a escrita da imagem: " << file_name << endl;
		status = 0;
	}

	// Alocação da estrutura de escrita:
	if (status)
	{
		image_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
		if (image_ptr == NULL)
		{
			cerr << "Falha ao alocar uma estrutura de escrita." << endl;
			status = 0;
		}
	}

	// Alocação da estrutura de meta-dados (informações complementares do arquivo):
	if (status)
	{
		info_ptr = png_create_info_struct(image_ptr);
		if (info_ptr == NULL)
		{
			cerr << "Falha ao alocar uma estrutura de meta-dados para a imagem." << endl;
			status = 0;
		}
	}
	// Criação da imagem:
	if (status)
	{
		if (setjmp(png_jmpbuf(image_ptr)))
		{
			cerr << "Erro durante a criação da imagem." << endl;
			status = 0;
		}
	}

	// Povoamento do arquivo, tanto com as informações complementares quanto com as cores:
	if (status)
	{
		png_init_io(image_ptr, file);
		png_set_IHDR(image_ptr, info_ptr, w, h,
								 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
								 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
		png_write_info(image_ptr, info_ptr);

		buffer_row = (png_bytep)malloc(3 * w * sizeof(png_byte));
		int y, x;
		for (y = 0; y < h; y++)
		{
			for (x = 0; x < w; x++)
			{
				setColorValue(&(buffer_row[x * 3]), buffer_image[y * w + x]);
			}
			png_write_row(image_ptr, buffer_row); // A trasferência do buffer para a imagem é feita linha por linha.
		}
		png_write_end(image_ptr, NULL);
	}

	// Em qualquer caso todas as estruturas criadas são fechadas e finalizadas:
	if (file != NULL)
		fclose(file);
	if (info_ptr != NULL)
		png_free_data(image_ptr, info_ptr, PNG_FREE_ALL, -1);
	if (image_ptr != NULL)
		png_destroy_write_struct(&image_ptr, (png_infopp)NULL);
	if (buffer_row != NULL)
		free(buffer_row);

	return status - 1;
}

// Função auxiliar que encontra o valor máximo num array de floats.
// (Utilizada para a normaização do buffer antes da geração da imagem)
float maximize(float *array, int array_size)
{
	float max = 757.0;

	for (int i = 0; i < array_size; i++)
	{
		if (array[i] > max)
		{
			max = array[i];
		}
	}
	return max;
}