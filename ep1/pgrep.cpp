#include <cstring>
#include <dirent.h>
#include <iostream>
#include <pthread.h>
#include <vector>
using namespace std;

void pgrep() {
    cout << "Pão\n";
}

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
    vector<string> v;
 
    MAX_THREADS = atoi(argv[1]);
    REGEX = argv[2];
    PATH = argv[3];

    cout << "MAX_THREADS: " << MAX_THREADS << "\n";
    cout << "REGEX: " << REGEX << "\n";
    cout << "PATH: " << PATH << "\n";
    
    pgrep();
    get_files(PATH, v, false);

    for (std::vector<string>::const_iterator i = v.begin(); i != v.end(); ++i)
        cout << *i << endl;
    
    return 0;
}