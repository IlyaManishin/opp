#pragma once 

typedef struct {
    int n;     
    double *A; 
    double *b;
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


