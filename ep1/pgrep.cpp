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
struct thr_data{
    vector<int>* indexes_ptr;
    vector<vector<string>*>* findings_ptr;
    vector<string>* files_ptr;
};

void pgrep() {
    cout << "Pão\n";
}

void *thr_func(void* arg) {
    thr_data* data= (struct thr_data*) arg;
    pthread_mutex_lock(&lock_cout);
    pthread_mutex_lock(&lock_indexes);
    int work_index=data->indexes_ptr->at(0);
    data->findings_ptr->at(0)->push_back("Wi");
    cout <<  data->findings_ptr->at(0)->back() << endl;
    cout << data->files_ptr->at(0) << endl;
    cout << "Olá eu sou uma thread! Eu estou viva e em breve morrerei." << endl;
    pthread_mutex_unlock(&lock_cout);
    pthread_mutex_unlock(&lock_indexes);

    pthread_exit(NULL);
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

    pthread_t thr[MAX_THREADS];
    int thr_data=0;
    int rc;

    if (files.size()<MAX_THREADS){
        MAX_THREADS=files.size();}



    struct thr_data data;
    data.indexes_ptr=&indexes;
    data.findings_ptr=&findings;
    data.files_ptr=&files;

    for(int i=0;i<MAX_THREADS;i++){
        if ((rc = pthread_create(&thr[i], NULL, thr_func, &data))) {
            fprintf(stderr, "error: pthread_create, rc: %d\n", rc);
            return EXIT_FAILURE;
        }
    } 

    for(int i=0;i<MAX_THREADS;i++){
        pthread_join(thr[i],NULL);
    }
    
    // string line="oi";
    // findings.at(0)->push_back(line);
    // for (std::vector<vector<string>*>::const_iterator i = findings.begin(); i != findings.end(); ++i){
    //     cout << (**i).size() << endl;



    // }



    
    return 0;
}