#include <cstring>
#include <dirent.h>
#include <iostream>
#include <pthread.h>
#include <vector>
#include <regex.h>
using namespace std;


/*VARIÁVEIS GLOBAIS*/

/* Declaração dos Mutexes. Eles são declarados como variáveis globais para que possam ser acessados de todas as Threads. */
pthread_mutex_t lock_indexes; //Mutex dos vetores de índices, para que nenhuma thread processe o mesmo arquivo duas vezes.
pthread_mutex_t lock_cout; //Mutex da saída de impressão padrão, para que threads não misturem o conteúdo de impressão de diferentes arquivos.

/* Estrutura de dados para encapsular de forma prática todos os dados que todas as threads precisam receber.
   A Struct é declarada gobalmente para que a função main e as demais threads possam acessá-la. */
struct thr_data{
    vector<int>* indexes_ptr; //Ponteiro para o vetor de índices.
    vector<vector<string>*>* findings_ptr; //Ponteiro para o vetor que armazena os ponteiros dos vetores correspondentes aos achados de cada arquivo.
    vector<string>* files_ptr; //Ponteiro do vetor que carrega os nomes dos arquivos a serem processados.
};

/*FUNÇÕES AUXILIARES*/

/* Função que lê o diretório passado pelo usuário, salvando o nome dos arquivos que precisam ser processados num vetor de strings
   TODO: É PRECISO ADICIONAR UMA VARIÁVEL DE ENTRADA QUE DEFINA SE A BUSCA VAI SER RECURSIVA OU NÃO... SE SIM EA PRECISA RETORNAR AS FILES
   DE TODOS OS SUBDIRETÓRIOS TAMBÉM, COM SEUS CAMINHOS RELATIVOS. DO JEITO QUE ESTÁ ELE NÃO TEM ESSA OPÇÃO E SÓ ESTÁ RETORNANDO NO VETOR
   OS NOMES DOS ARQUIVOS CONTIDOS NO DIRETORIO PRINCIPAL.
*/
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

/* FUNÇÃO DE THREAD */

/*Essa função define o que cada thread fará quando chamada. A idéia é que as threads sejam multi-uso e não voltadas a tarefas específicas.
  De forma que todas as threads são "iguais" e, nesse caso, só uma função de threads é necessária.*/
void *thr_func(void* arg) {
    thr_data* data= (struct thr_data*) arg; //Primeiro faz o cast do argumento para a struct apropriada. O Pthreads exige que os argumentos sejam dados como ponteiro de void, o que exige que façamos uma conversão de volta para o formato apropriado dentro da função de thread.
    
    pthread_mutex_lock(&lock_cout);
    cout << "Um dia eu farei algo útil, mas até lá, só serei feliz." << endl;
    pthread_mutex_unlock(&lock_cout);

    pthread_exit(NULL); //A thread é terminada com sinal nulo, que indica sucesso.
}


/* FUNÇÃO PRINCIPAL */
/* TODO: É PRECISO ALTERAR A LEITURA DOS ARGUMENTOS DE FORMA QUE EA RECEBA UM ARGUMENTO OPCIONAL QUE DEFINE SE A BUSCA DEVE SER RECURSIVA OU NÃO. ESSE ARGUMENTO DEVE ENTÃO SER REPASSADO PARA A FUNÇÃO DE LEITURA DE ARQUIVOS. */
int main(int argc, char *argv[]) {
    if (argc < 4) {
        cout << "usage: pgrep <MAX_THREADS> <REGEX_PESQUISA> <CAMINHO_DO_DIRETORIO>\n";
        return 1;
    }

    //Declaram-se e instanciam-se as variáveis de entrada:
    int MAX_THREADS;
    char *REGEX, *PATH;
    
    MAX_THREADS = atoi(argv[1]);
    REGEX = argv[2];
    PATH = argv[3];

    //Declaram-se as'estruturas de dados necessárias para o processamento das threads:
    vector<string> files;
    vector<int> indexes;
    vector<vector<string>*> findings;
    
    //Recolhem-se os nomes dos aqruivos a serem processados, colocando-os no vetor files:
    get_files(PATH, files, false);

    //Os vatores de índices e achados são povoados, respectivamente com índices e com ponteiros para vetores de saídas, um correspondente a cada arquivo:
    int index=0;
    for (std::vector<string>::const_iterator i = files.begin(); i != files.end(); ++i){
        indexes.push_back(index++);
        findings.push_back(new vector<string>);
    }

    //Os ponteiros das estruturas recém-criadas e iniciaizadas são encapsulados numa estrurura apropriada:
    struct thr_data data;
    data.indexes_ptr=&indexes;
    data.findings_ptr=&findings;
    data.files_ptr=&files;


    //São feitos os preparativos que permitem o processamento paralelo:
    if (files.size()<MAX_THREADS){MAX_THREADS=files.size();} //O Número de Threads é atuaizado, para que não sejam criadas threads ociosas.
    pthread_t thr[MAX_THREADS];// Cria-se o vetor que armazenará os ponteiros das threads.
    //Os Mutexes são inicializados:
    pthread_mutex_init(&lock_indexes,NULL);
    pthread_mutex_init(&lock_cout,NULL);


    //As Threads são criadas e cada uma delas invoca a função de thread com os dados contidos em data. Caso a criação de alguma thread
    // falhe, o programa retorna um erro e é encerrado.
    //TODO: TALVEZ FOSSE PRUDENTE MUDAR ISSO PARA QUE O PROGRAMA NÃO PARASSE CASO HOVESSE UMA FALHA NA CRIAÇÃO DE UMA THREAD. O IDEAL SERIA ATUALIZAR O NÚMERO DE MAX_THREADS, DECREMENTAR O i E NÃO RETORNAR FALHA. DE FORMA EUQ O PROGRAMA AINDA CONTINUARÁ RODANDO, MAS APENAS COM O NÚMERO DE THREADS QUE ELE FOI CAPAZ DE CRIAR. 
    int resp_creation;
    for(int i=0;i<MAX_THREADS;i++){
        if ((resp_creation = pthread_create(&thr[i], NULL, thr_func, &data))) {
            cerr << "error: attempt to create thread" << i << "failed! please check the capabilities of your system before defining an absurd number of threads.." << endl;
            return EXIT_FAILURE;
        }
    } 

    //Esse join faz com que o programa só continue, ou seja, seja encerrado, quando todas as threads tenham retornado com sucesso:
    for(int i=0;i<MAX_THREADS;i++){
        pthread_join(thr[i],NULL);
    }
    
    return 0; //Retorna-se um 0 bem rechonchudo e saboroso.
}


//Uns lixinhos que o sub quer guardar por enquanto pra consultar depois:

    // string line="oi";
    // findings.at(0)->push_back(line);
    // for (std::vector<vector<string>*>::const_iterator i = findings.begin(); i != findings.end(); ++i){
    //     cout << (**i).size() << endl;
    // }

    //thrfunc:
    // pthread_mutex_lock(&lock_cout);
    // pthread_mutex_lock(&lock_indexes);
    // int work_index=data->indexes_ptr->at(0);
    // data->findings_ptr->at(0)->push_back("Wi");
    // cout <<  data->findings_ptr->at(0)->back() << endl;
    // cout << data->files_ptr->at(0) << endl;
    // cout << "Olá eu sou uma thread! Eu estou viva e em breve morrerei." << endl;
    // pthread_mutex_unlock(&lock_cout);
    // pthread_mutex_unlock(&lock_indexes);