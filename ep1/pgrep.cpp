#include <cstring>
#include <dirent.h>
#include <iostream>
#include <pthread.h>
#include <vector>
#include <regex.h>
using namespace std;


//Os Mutexes são decarados como variáveis globais de forma que eles fiquem acessives para todas as threads:
pthread_mutex_t lock_indexes;
pthread_mutex_t lock_cout;

void pgrep() {
    cout << "Pão\n";
}


//Falta adicionar um modo recursivo:
void get_files(const string &path, vector<string> &files, const bool show_hidden = false){
    DIR *dir;
    struct dirent *epdf;
    dir = opendir(path.c_str());

    if (dir != NULL){
        while ((epdf = readdir(dir)) != NULL){
            if(show_hidden ? (epdf->d_type==DT_DIR && string(epdf->d_name) != ".." && string(epdf->d_name) != "." ) : (epdf->d_type==DT_DIR && strstr(epdf->d_name,"..") == NULL && strstr(epdf->d_name,".") == NULL ) ){
                get_files(path+epdf->d_name+"/",files, show_hidden);
            }
            if(epdf->d_type==DT_REG){
                files.push_back(path+epdf->d_name);
            }
        }
    }
    closedir(dir);
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        cout << "usage: pgrep <MAX_THREADS> <REGEX_PESQUISA> <CAMINHO_DO_DIRETORIO>\n";
        return 1;
    }

    int MAX_THREADS;
    char *REGEX, *PATH;
    vector<string> files;
    vector<int> indexes;
    vector<vector<string>*> findings;

 
    MAX_THREADS = atoi(argv[1]);
    REGEX = argv[2];
    PATH = argv[3];

    cout << "MAX_THREADS: " << MAX_THREADS << "\n";
    cout << "REGEX: " << REGEX << "\n";
    cout << "PATH: " << PATH << "\n";

    get_files(PATH, files, false);
    int cont=0;
    for (std::vector<string>::const_iterator i = files.begin(); i != files.end(); ++i){
        cout << cont;
        indexes.push_back(cont++);
        findings.push_back(new vector<string>);
        cout << *i << endl;
    }
    pthread_mutex_init(&lock_indexes,NULL);
    pthread_mutex_init(&lock_cout,NULL);
    
    // string line="oi";
    // findings.at(0)->push_back(line);
    // for (std::vector<vector<string>*>::const_iterator i = findings.begin(); i != findings.end(); ++i){
    //     cout << (**i).size() << endl;



    // }



    
    return 0;
}