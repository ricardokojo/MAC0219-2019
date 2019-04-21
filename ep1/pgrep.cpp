#include <cstring>
#include <dirent.h>
#include <iostream>
#include <pthread.h>
#include <vector>
#include <regex.h>
#include <fstream>
#include <unistd.h>
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
    regex_t* preg_ptr; //Ponteiro para o objeto que representa a regex compilada.
};

/*FUNÇÕES AUXILIARES*/

/* Função que lê o diretório passado pelo usuário, salvando o nome dos arquivos que precisam ser processados num vetor de strings
   TODO: É NECESSÁRIO PODER CONTROLAR QUANDO A BUSCA É RECURSIVA QUANDO NÃO, POR QUE ISSO DEVERIA SER OPÇÃO DO USUARIO.
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
    //cada thread cuida de um arquivo por vez... Caso o numero de arquivos seja maior que  o de threads, elas ficam percorrendo os aqruivos livres em loop:
    int time2work=1; //Indica que ainda há arquivos a serem buscados.
    while(time2work){
        pthread_mutex_lock(&lock_indexes); //Usa-se o mutex para consultar os indices dos arquivos a serem processados, sem gerar condição de corrida.
        //Se o vetor de indices esta vazio não há o que fazer:
        if(data->indexes_ptr->empty()){
            pthread_mutex_unlock(&lock_indexes);
            time2work=0;
        }
        //Caso contrário o a thread pega seu indice de trabalho e desbloqueia o vetor de indices:
        else{
            int work_index=data->indexes_ptr->back(); //Pega-se o indice
            data->indexes_ptr->pop_back();//Remove o indice do vetor para que nenhuma outra thread vote a pegá-lo.
            pthread_mutex_unlock(&lock_indexes);
            ifstream work_file(data->files_ptr->at(work_index)); //abre-se o leitor do arquivo correspondente ao indice de trabalho
            string line;
            int line_cont=0; //Contador de linhas para marcar as linhas em que há match com a regex.
            // Percorre-se o arquivo lendo linha por linha e procurando matches com a regex:
            while(!work_file.eof()){
                work_file >> line;
                const char* c_line=line.c_str(); // É necessario converter a string para um ponteiro de char, já que o regex.h foi feito para trabalhar com strings em c.
                //Verifica-se a compatibilidade, caso haja match adiciona-se o resultado ao vetor de findings referente aquele arquivo.
                if (regexec(data->preg_ptr, c_line, 0, 0, 0)==0){
                    data->findings_ptr->at(work_index)->push_back(data->files_ptr->at(work_index)+":"+to_string(line_cont));
                }
                line_cont++;
            }

            //Imprime os matches daquele arquivo. Para isso é necessário trancar a saída.
            pthread_mutex_lock(&lock_cout);
            while(!data->findings_ptr->at(work_index)->empty()){
                cout << data->findings_ptr->at(work_index)->back() << endl;
                data->findings_ptr->at(work_index)->pop_back(); //Remove-se a linha impressa do vetor
            }
            pthread_mutex_unlock(&lock_cout);

        }
    }
    pthread_exit(NULL);// Ao final a thread é encerrada com resultado Nulo, o que indica seu sucesso.
}



/* FUNÇÃO PRINCIPAL */
/* TODO: É PRECISO ALTERAR A LEITURA DOS ARGUMENTOS DE FORMA QUE    ELA RECEBA UM ARGUMENTO OPCIONAL QUE DEFINE SE A BUSCA DEVE SER RECURSIVA OU NÃO. ESSE ARGUMENTO DEVE ENTÃO SER REPASSADO PARA A FUNÇÃO DE LEITURA DE ARQUIVOS. */
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


    //Os valores de índices e achados são povoados, respectivamente com índices e com ponteiros para vetores de saídas, um correspondente a cada arquivo:
    int index=0;
    for (std::vector<string>::const_iterator i = files.begin(); i != files.end(); ++i){
        indexes.push_back(index++);
        findings.push_back(new vector<string>);
    }

    //Declaram-se as variáveis necessárias para a compilação da regex.
    regex_t preg;
    int resp_regex_comp;

    //Compila-se a regex de entrada. Como só existe uma única regex de busca, ela só precisa ser compilada uma única vez e sua versão compilada pode ser passada para todas a threads para comparação, economizando processamento:
    //WARNING:AS CINCO LINHAS DE CÓDIGO A SEGUIR NÃO FORAM APROPRIADAMENTE TESTADAS AINDA. PROSSIGA COM CAUTELA.
    resp_regex_comp=regcomp(&preg,REGEX,0);
    if(resp_regex_comp!=0){
        cerr << "error: regex compilation failed; please, verify you regex." << endl;
        return EXIT_FAILURE; //Caso a regex não possa ser compilada com sucesso o programa é encerrado, pois é impossível continuar assim.
    }

    //Os ponteiros das estruturas recém-criadas e iniciaizadas são encapsulados numa estrurura apropriada:
    struct thr_data data;
    data.indexes_ptr=&indexes;
    data.findings_ptr=&findings;
    data.files_ptr=&files;
    data.preg_ptr=&preg;


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