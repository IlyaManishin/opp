#pragma once 

typedef struct {
    int n;     
    double *A; 
    double *b;
} LinearSystem;

LinearSystem read_lin_system(const char *filename);

