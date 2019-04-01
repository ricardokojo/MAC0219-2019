#include <cstring>
#include <dirent.h>
#include <iostream>
#include <vector>
#include <regex.h>
using namespace std;



int match(string text, string pattern){

	regex_t preg;
	int resp_comp;
	int resp_exec;
	const char *ctext = text.c_str();
	const char *cpattern = pattern.c_str();
	resp_comp=regcomp(&preg,cpattern,0);
	if(resp_comp!=0){
		printf("ERRO DE COMPILAÇÃO");
		return(0);
	}

	else{
		resp_exec=regexec(&preg, ctext, 0, 0, 0);
		return(resp_exec);
	}

}

int main(){
	string text= "a very simple simple simple string";
	string pattern="b\\{1,2\\}";
	int resp=match(text,pattern);
	printf("Resposta: %d", resp);
	return(resp);
}