#pragma once 

typedef struct {
    int n;     
    double *A; 
    int A_rows_count;

    double *b;
    int b_rows_count;
    
    double *r;
} TLinearSystem;

typedef struct {
    int A_StartRow;
    int A_EndRow;
    int b_Start;
    int b_End;
} TLoadRange;

int get_lin_system_size(const char *filename);
TLinearSystem read_lin_system(const char *filename, TLoadRange range);
void free_lin_system(TLinearSystem *sys);

void writeAnswer(const char* destPath, double *x, int n);


