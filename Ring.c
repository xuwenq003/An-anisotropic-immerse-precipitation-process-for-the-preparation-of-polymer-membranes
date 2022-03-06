//
//  main.c
//  phase field simulation in a ring
//
//  Created by Mac on 2019/5/21.
//  Copyright © 2019 Mac. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define NUMTHREAD   8

typedef struct
{
    int row;   //矩阵行数
    int col;   //矩阵列数
    double ** data;   //二维数组作为矩阵
}
matrix;

//变量
/*physical properties*/
double M_ss,M_sp,M_ps,K_pp,K_ss,mp;
double M_ppr, M_pps;

/*geometry*/
double Lr,Ls,Ri,Ro,Ifi,Ifo,dr,ds,dt,val_cut;
int Nr,Ns;

/*grids*/
matrix *rc,*rf,*urlap,*uslap;

/*concentration*/
matrix *phi_s,*phi_p;

/*other field*/
matrix *chp_p,*chp_s,*dPsi_ds,*dPsi_dp,*coefficient_r,*coefficient_s,*hexp_s,*hexp_p,*himp_s,*himp_p;
matrix *hexp_s_np,*hexp_p_np;

/*initial concentration*/
matrix *phi_p_ini,*phi_s_ini;

/*创建矩阵*/
void matrix_create(int row,int col,matrix *mat)
{
    int i=0;
    mat->row=row;
    mat->col=col;
    mat->data=(double **)malloc(row*sizeof(double *));
    #pragma omp parallel for num_threads(NUMTHREAD)
    for (i=0;i<row;i++)
    {
        mat->data[i]=(double *)malloc(col*sizeof(double));
    }
}

/*生成全“0”矩阵*/
void matrix_zeros(int row,int col,matrix *mat)
{
    int i=0,j=0;
    matrix_create(row,col,mat);
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<row;i++)
        for(j=0;j<col;j++)
            mat->data[i][j]=0.0;
}

/*生成全“1”矩阵*/
void matrix_ones(int row,int col,matrix *mat)
{
    int i=0,j=0;
    matrix_create(row,col,mat);
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<row;i++)
        for(j=0;j<col;j++)
            mat->data[i][j]=1.0;
}

/*删除矩阵*/
void matrix_free(matrix *mat)
{
    int i=0;
    #pragma omp parallel for num_threads(NUMTHREAD)
    for (i=0;i<mat->row;i++)
    {
        free (mat->data[i]);
    }
    free (mat->data);
}

/*矩阵相加*/
void matrix_add(matrix *mat1,matrix *mat2,matrix *matresult)
{
    int i=0,j=0;
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<matresult->row;i++)
        for(j=0;j<matresult->col;j++)
            matresult->data[i][j]=mat1->data[i][j]+mat2->data[i][j];
}

/*矩阵相减*/
void matrix_minus(matrix *mat1,matrix *mat2,matrix *matresult)
{
    int i=0,j=0;
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<matresult->row;i++)
        for(j=0;j<matresult->col;j++)
            matresult->data[i][j]=mat1->data[i][j]-mat2->data[i][j];
}

/*矩阵数乘*/
void matrix_multiply(double coefficient,matrix *mat,matrix *matresult)
{
    int i=0,j=0;
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<mat->row;i++)
        for(j=0;j<mat->col;j++)
            matresult->data[i][j]=mat->data[i][j]*coefficient;
}

/*矩阵乘法（点乘）*/
void matrix_dotproduct(matrix *mat1,matrix *mat2,matrix *matresult)
{
    int i=0,j=0,k=0;
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<mat1->row;i++)
        for(j=0;j<mat2->col;j++)
            for(k=0;k<mat1->col;k++)
                matresult->data[i][j]+=mat1->data[i][k]+mat2->data[k][j];
}


/*矩阵转置*/
void matrix_transpose(matrix *mat,matrix *mat_t)
{
    int i=0,j=0;
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<mat->row;i++)
        for(j=0;j<mat->col;j++)
            mat_t->data[j][i]=mat->data[i][j];
}

/*矩阵平铺*/
void matrix_replication(matrix *mat,matrix *matresult,int rowrep,int colrep)
{
    int i=0,j=0;
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<mat->row*rowrep;i++)
        for(j=0;j<mat->col*colrep;j++)
            matresult->data[i][j]=mat->data[i%mat->row][j%mat->col];
}

/*矩阵打印*/
void matrix_print(matrix *mat)
{
    int i=0,j=0;
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<mat->row;i++)
    {
        for(j=0;j<mat->col;j++)
        {
            printf("%g ",mat->data[i][j]);
        }
        printf("\n");
    }
}
/*求矩阵中元素平均值*/
double matrix_average(matrix *mat)
{
    int i=0,j=0;
    double average=0;
    for(i=0;i<mat->row;i++)
        for(j=0;j<mat->col;j++)
            average+=mat->data[i][j];
    average=average/(mat->col*mat->row);
    return average;
}

/*求矩阵中元素标准差*/
double matrix_standarderror(matrix *mat)
{
    int i=0,j=0;
    double std=0;
    double average=0;
    average=matrix_average(mat);
    for(i=0;i<mat->row;i++)
        for(j=0;j<mat->col;j++)
        {
            std=std+(mat->data[i][j]-average)*(mat->data[i][j]-average);
        }
    std=sqrt(std);
    return std;
}

/*求矩阵元素最大值*/
double matrix_max(matrix *mat)
{
    int i=0,j=0;
    double max=0;
    for(i=0;i<mat->row;i++)
    {
        for(j=0;j<mat->col;j++)
        {
            if(mat->data[i][j]>max)
                max=mat->data[i][j];
        }
    }
    return max;
}


/*求矩阵的均方根*/
double matrix_root_mean_square(matrix *mat)
{
    int i=0,j=0;
    double root_mean_square=0;
    for(i=0;i<mat->row;i++)
    for(j=0;j<mat->col;j++)
    root_mean_square+=(mat->data[i][j]*mat->data[i][j]);
    root_mean_square=sqrt(root_mean_square/(mat->col*mat->row));
    return root_mean_square;
}


/*拉普拉斯算子p*/
void laplacian_p(matrix *fin,matrix *fout)
{
    int i=0,j=0;
    matrix *fin_all=0;
    fin_all=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nr+2,Ns+2,fin_all);
    //中间部分
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<Nr;i++)
    {
        for(j=0;j<Ns;j++)
        {
            fin_all->data[i+1][j+1]=fin->data[i][j];
        }
    }
    //边界
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=1;i<Nr+1;i++)    //s方向周期边界条件
    {
        fin_all->data[i][0]=fin_all->data[i][Ns];
        fin_all->data[i][Ns+1]=fin_all->data[i][1];
    }
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(j=0;j<Ns+2;j++)      //r方向无穿透边界条件
    {
        fin_all->data[Nr+1][j]=fin_all->data[Nr][j];
        fin_all->data[0][j]=fin_all->data[1][j];
    }
    
    //利用差分表达拉普拉斯
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<Nr;i++)
        for(j=0;j<Ns;j++)
            fout->data[i][j]=
            M_ppr*(fin_all->data[i+2][j+1]*rc->data[i+2][0]-2.0*fin_all->data[i+1][j+1]*rf->data[i+1][0]+fin_all->data[i][j+1]*rc->data[i+1][0])*urlap->data[i][j]           +M_pps*(fin_all->data[i+1][j+2]-2.0*fin_all->data[i+1][j+1]+fin_all->data[i+1][j])*uslap->data[i][j];
    
    matrix_free(fin_all);
}

/*二次拉普拉斯算子p*/
void doublelaplacian_p(matrix* fin,matrix* fout)
{
    int i=0,j=0;
    matrix *fin_all=0;
    fin_all=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nr+4,Ns+4,fin_all);
    //中间部分
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<Nr;i++)
    {
        for(j=0;j<Ns;j++)
        {
            fin_all->data[i+2][j+2]=fin->data[i][j];
        }
    }
    //边界
    //s方向周期边界条件
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=2;i<=Nr+1;i++)
    {
        fin_all->data[i][0]=fin_all->data[i][Ns];
        fin_all->data[i][1]=fin_all->data[i][Ns+1];
        fin_all->data[i][Ns+2]=fin_all->data[i][2];
        fin_all->data[i][Ns+3]=fin_all->data[i][3];
    }
    //r方向无穿透边界条件
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(j=0;j<=Ns+3;j++)
    {
        //r方向边界phi的法向导数为0
        fin_all->data[Nr+2][j]=fin_all->data[Nr+1][j];
        fin_all->data[1][j]=fin_all->data[2][j];
        
        //r方向边界拉普拉斯phi的法向导数为0
    fin_all->data[Nr+3][j]=(rf->data[Nr+1][0]/rf->data[Nr][0]*((rc->data[Nr+1][0]-2*rf->data[Nr][0])*fin_all->data[Nr+1][j]+rc->data[Nr][0]*fin_all->data[Nr][j])-(rc->data[Nr+1][0]-2*rf->data[Nr+1][0])*fin_all->data[Nr+1][j]+dr*dr/ds/ds*rf->data[Nr+1][0]*(1.0/rf->data[Nr][0]/rf->data[Nr][0]-1.0/rf->data[Nr+1][0]/rf->data[Nr+1][0])*(fin_all->data[Nr+1][j+1]-2*fin_all->data[Nr+1][j]+fin_all->data[Nr+1][j-1]))/rc->data[Nr+2][0];
    fin_all->data[0][j]=(rf->data[0][0]/rf->data[1][0]*(rc->data[2][0]*fin_all->data[3][j]+(rc->data[1][0]-2*rf->data[1][0])*fin_all->data[2][j])-(rc->data[1][0]-2*rf->data[0][0])*fin_all->data[2][j]+dr*dr/ds/ds*rf->data[0][0]*(1.0/rf->data[1][0]/rf->data[1][0]-1.0/rf->data[0][0]/rf->data[0][0])*(fin_all->data[2][j+1]-2*fin_all->data[2][j]+fin_all->data[2][j-1]))/rc->data[0][0];
                                                                                                    
    }
    //利用差分表达二次拉普拉斯
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<Nr;i++)
        for(j=0;j<Ns;j++)
            fout->data[i][j]=
            M_ppr*(rc->data[i+2][0]*((rc->data[i+3][0]*fin_all->data[i+4][j+2]-2*rf->data[i+2][0]*fin_all->data[i+3][j+2]+rc->data[i+2][0]*fin_all->data[i+2][j+2])/rf->data[i+2][0]/dr/dr+(fin_all->data[i+3][j+3]-2*fin_all->data[i+3][j+2]+fin_all->data[i+3][j+1])/rf->data[i+2][0]/rf->data[i+2][0]/ds/ds)
            -2*rf->data[i+1][0]*((rc->data[i+2][0]*fin_all->data[i+3][j+2]-2*rf->data[i+1][0]*fin_all->data[i+2][j+2]+rc->data[i+1][0]*fin_all->data[i+1][j+2])/rf->data[i+1][0]/dr/dr+(fin_all->data[i+2][j+3]-2*fin_all->data[i+2][j+2]+fin_all->data[i+2][j+1])/rf->data[i+1][0]/rf->data[i+1][0]/ds/ds)
            +rc->data[i+1][0]*((rc->data[i+1][0]*fin_all->data[i+2][j+2]-2*rf->data[i][0]*fin_all->data[i+1][j+2]+rc->data[i][0]*fin_all->data[i][j+2])/rf->data[i][0]/dr/dr+(fin_all->data[i+1][j+3]-2*fin_all->data[i+1][j+2]+fin_all->data[i+1][j+1])/rf->data[i][0]/rf->data[i][0]/ds/ds))*urlap->data[i][j]
            +M_pps*((rc->data[i+2][0]*fin_all->data[i+3][j+3]-2*rf->data[i+1][0]*fin_all->data[i+2][j+3]+rc->data[i+1][0]*fin_all->data[i+1][j+3])/rf->data[i+1][0]/dr/dr+(fin_all->data[i+2][j+4]-2*fin_all->data[i+2][j+3]+fin_all->data[i+2][j+2])/rf->data[i+1][0]/rf->data[i+1][0]/ds/ds
            -2*((rc->data[i+2][0]*fin_all->data[i+3][j+2]-2*rf->data[i+1][0]*fin_all->data[i+2][j+2]+rc->data[i+1][0]*fin_all->data[i+1][j+2])/rf->data[i+1][0]/dr/dr+(fin_all->data[i+2][j+3]-2*fin_all->data[i+2][j+2]+fin_all->data[i+2][j+1])/rf->data[i+1][0]/rf->data[i+1][0]/ds/ds)
            +(rc->data[i+2][0]*fin_all->data[i+3][j+1]-2*rf->data[i+1][0]*fin_all->data[i+2][j+1]+rc->data[i+1][0]*fin_all->data[i+1][j+1])/rf->data[i+1][0]/dr/dr+(fin_all->data[i+2][j+2]-2*fin_all->data[i+2][j+1]+fin_all->data[i+2][j])/rf->data[i+1][0]/rf->data[i+1][0]/ds/ds
                )*uslap->data[i][j];
    
    matrix_free(fin_all);
    
}

/*去中心的二次拉普拉斯算子p*/
void doublelaplacian_withoutcenter_p(matrix* fin,matrix* fout)
{
    int i=0,j=0;
    matrix *fin_all=0;
    fin_all=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nr+4,Ns+4,fin_all);
    //中间部分
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<Nr;i++)
    {
        for(j=0;j<Ns;j++)
        {
            fin_all->data[i+2][j+2]=fin->data[i][j];
        }
    }
    //边界
    //s方向周期边界条件
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=2;i<=Nr+1;i++)
    {
        fin_all->data[i][0]=fin_all->data[i][Ns];
        fin_all->data[i][1]=fin_all->data[i][Ns+1];
        fin_all->data[i][Ns+2]=fin_all->data[i][2];
        fin_all->data[i][Ns+3]=fin_all->data[i][3];
    }
    //r方向无穿透边界条件
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(j=0;j<=Ns+3;j++)
    {
        //r方向边界phi的法向导数为0
        fin_all->data[Nr+2][j]=fin_all->data[Nr+1][j];
        fin_all->data[1][j]=fin_all->data[2][j];
        
        //r方向边界拉普拉斯phi的法向导数为0
    fin_all->data[Nr+3][j]=(rf->data[Nr+1][0]/rf->data[Nr][0]*((rc->data[Nr+1][0]-2*rf->data[Nr][0])*fin_all->data[Nr+1][j]+rc->data[Nr][0]*fin_all->data[Nr][j])-(rc->data[Nr+1][0]-2*rf->data[Nr+1][0])*fin_all->data[Nr+1][j]+dr*dr/ds/ds*rf->data[Nr+1][0]*(1.0/rf->data[Nr][0]/rf->data[Nr][0]-1.0/rf->data[Nr+1][0]/rf->data[Nr+1][0])*(fin_all->data[Nr+1][j+1]-2*fin_all->data[Nr+1][j]+fin_all->data[Nr+1][j-1]))/rc->data[Nr+2][0];
    fin_all->data[0][j]=(rf->data[0][0]/rf->data[1][0]*(rc->data[2][0]*fin_all->data[3][j]+(rc->data[1][0]-2*rf->data[1][0])*fin_all->data[2][j])-(rc->data[1][0]-2*rf->data[0][0])*fin_all->data[2][j]+dr*dr/ds/ds*rf->data[0][0]*(1.0/rf->data[1][0]/rf->data[1][0]-1.0/rf->data[0][0]/rf->data[0][0])*(fin_all->data[2][j+1]-2*fin_all->data[2][j]+fin_all->data[2][j-1]))/rc->data[0][0];
                                                                                                    
    }
    //利用差分表达二次拉普拉斯(去掉中心点)
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<Nr;i++)
        for(j=0;j<Ns;j++)
            fout->data[i][j]=
            M_ppr*(rc->data[i+2][0]*((rc->data[i+3][0]*fin_all->data[i+4][j+2]-2*rf->data[i+2][0]*fin_all->data[i+3][j+2])/rf->data[i+2][0]/dr/dr+(fin_all->data[i+3][j+3]-2*fin_all->data[i+3][j+2]+fin_all->data[i+3][j+1])/rf->data[i+2][0]/rf->data[i+2][0]/ds/ds)
            -2*rf->data[i+1][0]*((rc->data[i+2][0]*fin_all->data[i+3][j+2]+rc->data[i+1][0]*fin_all->data[i+1][j+2])/rf->data[i+1][0]/dr/dr+(fin_all->data[i+2][j+3]+fin_all->data[i+2][j+1])/rf->data[i+1][0]/rf->data[i+1][0]/ds/ds)
            +rc->data[i+1][0]*((-2*rf->data[i][0]*fin_all->data[i+1][j+2]+rc->data[i][0]*fin_all->data[i][j+2])/rf->data[i][0]/dr/dr+(fin_all->data[i+1][j+3]-2*fin_all->data[i+1][j+2]+fin_all->data[i+1][j+1])/rf->data[i][0]/rf->data[i][0]/ds/ds))*urlap->data[i][j]
            +M_pps*((rc->data[i+2][0]*fin_all->data[i+3][j+3]-2*rf->data[i+1][0]*fin_all->data[i+2][j+3]+rc->data[i+1][0]*fin_all->data[i+1][j+3])/rf->data[i+1][0]/dr/dr+(fin_all->data[i+2][j+4]-2*fin_all->data[i+2][j+3])/rf->data[i+1][0]/rf->data[i+1][0]/ds/ds
            -2*((rc->data[i+2][0]*fin_all->data[i+3][j+2]+rc->data[i+1][0]*fin_all->data[i+1][j+2])/rf->data[i+1][0]/dr/dr+(fin_all->data[i+2][j+3]+fin_all->data[i+2][j+1])/rf->data[i+1][0]/rf->data[i+1][0]/ds/ds)
            +(rc->data[i+2][0]*fin_all->data[i+3][j+1]-2*rf->data[i+1][0]*fin_all->data[i+2][j+1]+rc->data[i+1][0]*fin_all->data[i+1][j+1])/rf->data[i+1][0]/dr/dr+(-2*fin_all->data[i+2][j+1]+fin_all->data[i+2][j])/rf->data[i+1][0]/rf->data[i+1][0]/ds/ds
                )*uslap->data[i][j];
    
    matrix_free(fin_all);
    
}




/*拉普拉斯算子*/
void laplacian(matrix *fin,matrix *fout)
{
    int i=0,j=0;
    matrix *fin_all=0;
    fin_all=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nr+2,Ns+2,fin_all);
    //中间部分
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<Nr;i++)
    {
        for(j=0;j<Ns;j++)
        {
            fin_all->data[i+1][j+1]=fin->data[i][j];
        }
    }
    //边界
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=1;i<Nr+1;i++)    //s方向周期边界条件
    {
        fin_all->data[i][0]=fin_all->data[i][Ns];
        fin_all->data[i][Ns+1]=fin_all->data[i][1];
    }
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(j=0;j<Ns+2;j++)      //r方向无穿透边界条件
    {
        fin_all->data[Nr+1][j]=fin_all->data[Nr][j];
        fin_all->data[0][j]=fin_all->data[1][j];
    }
    
    //利用差分表达拉普拉斯
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<Nr;i++)
        for(j=0;j<Ns;j++)
            fout->data[i][j]=
            (fin_all->data[i+2][j+1]*rc->data[i+2][0]-2.0*fin_all->data[i+1][j+1]*rf->data[i+1][0]+fin_all->data[i][j+1]*rc->data[i+1][0])*urlap->data[i][j]           +(fin_all->data[i+1][j+2]-2.0*fin_all->data[i+1][j+1]+fin_all->data[i+1][j])*uslap->data[i][j];
    
    matrix_free(fin_all);
}

/*二次拉普拉斯算子*/
void doublelaplacian(matrix* fin,matrix* fout)
{
    int i=0,j=0;
    matrix *fin_all=0;
    fin_all=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nr+4,Ns+4,fin_all);
    //中间部分
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<Nr;i++)
    {
        for(j=0;j<Ns;j++)
        {
            fin_all->data[i+2][j+2]=fin->data[i][j];
        }
    }
    //边界
    //s方向周期边界条件
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=2;i<=Nr+1;i++)
    {
        fin_all->data[i][0]=fin_all->data[i][Ns];
        fin_all->data[i][1]=fin_all->data[i][Ns+1];
        fin_all->data[i][Ns+2]=fin_all->data[i][2];
        fin_all->data[i][Ns+3]=fin_all->data[i][3];
    }
    //r方向无穿透边界条件
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(j=0;j<=Ns+3;j++)
    {
        //r方向边界phi的法向导数为0
        fin_all->data[Nr+2][j]=fin_all->data[Nr+1][j];
        fin_all->data[1][j]=fin_all->data[2][j];
        
        //r方向边界拉普拉斯phi的法向导数为0
    fin_all->data[Nr+3][j]=(rf->data[Nr+1][0]/rf->data[Nr][0]*((rc->data[Nr+1][0]-2*rf->data[Nr][0])*fin_all->data[Nr+1][j]+rc->data[Nr][0]*fin_all->data[Nr][j])-(rc->data[Nr+1][0]-2*rf->data[Nr+1][0])*fin_all->data[Nr+1][j]+dr*dr/ds/ds*rf->data[Nr+1][0]*(1.0/rf->data[Nr][0]/rf->data[Nr][0]-1.0/rf->data[Nr+1][0]/rf->data[Nr+1][0])*(fin_all->data[Nr+1][j+1]-2*fin_all->data[Nr+1][j]+fin_all->data[Nr+1][j-1]))/rc->data[Nr+2][0];
    fin_all->data[0][j]=(rf->data[0][0]/rf->data[1][0]*(rc->data[2][0]*fin_all->data[3][j]+(rc->data[1][0]-2*rf->data[1][0])*fin_all->data[2][j])-(rc->data[1][0]-2*rf->data[0][0])*fin_all->data[2][j]+dr*dr/ds/ds*rf->data[0][0]*(1.0/rf->data[1][0]/rf->data[1][0]-1.0/rf->data[0][0]/rf->data[0][0])*(fin_all->data[2][j+1]-2*fin_all->data[2][j]+fin_all->data[2][j-1]))/rc->data[0][0];
                                                                                                    
    }
    //利用差分表达二次拉普拉斯
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<Nr;i++)
        for(j=0;j<Ns;j++)
            fout->data[i][j]=
            (rc->data[i+2][0]*((rc->data[i+3][0]*fin_all->data[i+4][j+2]-2*rf->data[i+2][0]*fin_all->data[i+3][j+2]+rc->data[i+2][0]*fin_all->data[i+2][j+2])/rf->data[i+2][0]/dr/dr+(fin_all->data[i+3][j+3]-2*fin_all->data[i+3][j+2]+fin_all->data[i+3][j+1])/rf->data[i+2][0]/rf->data[i+2][0]/ds/ds)
            -2*rf->data[i+1][0]*((rc->data[i+2][0]*fin_all->data[i+3][j+2]-2*rf->data[i+1][0]*fin_all->data[i+2][j+2]+rc->data[i+1][0]*fin_all->data[i+1][j+2])/rf->data[i+1][0]/dr/dr+(fin_all->data[i+2][j+3]-2*fin_all->data[i+2][j+2]+fin_all->data[i+2][j+1])/rf->data[i+1][0]/rf->data[i+1][0]/ds/ds)
            +rc->data[i+1][0]*((rc->data[i+1][0]*fin_all->data[i+2][j+2]-2*rf->data[i][0]*fin_all->data[i+1][j+2]+rc->data[i][0]*fin_all->data[i][j+2])/rf->data[i][0]/dr/dr+(fin_all->data[i+1][j+3]-2*fin_all->data[i+1][j+2]+fin_all->data[i+1][j+1])/rf->data[i][0]/rf->data[i][0]/ds/ds))*urlap->data[i][j]
            +((rc->data[i+2][0]*fin_all->data[i+3][j+3]-2*rf->data[i+1][0]*fin_all->data[i+2][j+3]+rc->data[i+1][0]*fin_all->data[i+1][j+3])/rf->data[i+1][0]/dr/dr+(fin_all->data[i+2][j+4]-2*fin_all->data[i+2][j+3]+fin_all->data[i+2][j+2])/rf->data[i+1][0]/rf->data[i+1][0]/ds/ds
            -2*((rc->data[i+2][0]*fin_all->data[i+3][j+2]-2*rf->data[i+1][0]*fin_all->data[i+2][j+2]+rc->data[i+1][0]*fin_all->data[i+1][j+2])/rf->data[i+1][0]/dr/dr+(fin_all->data[i+2][j+3]-2*fin_all->data[i+2][j+2]+fin_all->data[i+2][j+1])/rf->data[i+1][0]/rf->data[i+1][0]/ds/ds)
            +(rc->data[i+2][0]*fin_all->data[i+3][j+1]-2*rf->data[i+1][0]*fin_all->data[i+2][j+1]+rc->data[i+1][0]*fin_all->data[i+1][j+1])/rf->data[i+1][0]/dr/dr+(fin_all->data[i+2][j+2]-2*fin_all->data[i+2][j+1]+fin_all->data[i+2][j])/rf->data[i+1][0]/rf->data[i+1][0]/ds/ds
                )*uslap->data[i][j];
    
    matrix_free(fin_all);
    
}

/*去中心的二次拉普拉斯算子*/
void doublelaplacian_withoutcenter(matrix* fin,matrix* fout)
{
    int i=0,j=0;
    matrix *fin_all=0;
    fin_all=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nr+4,Ns+4,fin_all);
    //中间部分
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<Nr;i++)
    {
        for(j=0;j<Ns;j++)
        {
            fin_all->data[i+2][j+2]=fin->data[i][j];
        }
    }
    //边界
    //s方向周期边界条件
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=2;i<=Nr+1;i++)
    {
        fin_all->data[i][0]=fin_all->data[i][Ns];
        fin_all->data[i][1]=fin_all->data[i][Ns+1];
        fin_all->data[i][Ns+2]=fin_all->data[i][2];
        fin_all->data[i][Ns+3]=fin_all->data[i][3];
    }
    //r方向无穿透边界条件
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(j=0;j<=Ns+3;j++)
    {
        //r方向边界phi的法向导数为0
        fin_all->data[Nr+2][j]=fin_all->data[Nr+1][j];
        fin_all->data[1][j]=fin_all->data[2][j];
        
        //r方向边界拉普拉斯phi的法向导数为0
    fin_all->data[Nr+3][j]=(rf->data[Nr+1][0]/rf->data[Nr][0]*((rc->data[Nr+1][0]-2*rf->data[Nr][0])*fin_all->data[Nr+1][j]+rc->data[Nr][0]*fin_all->data[Nr][j])-(rc->data[Nr+1][0]-2*rf->data[Nr+1][0])*fin_all->data[Nr+1][j]+dr*dr/ds/ds*rf->data[Nr+1][0]*(1.0/rf->data[Nr][0]/rf->data[Nr][0]-1.0/rf->data[Nr+1][0]/rf->data[Nr+1][0])*(fin_all->data[Nr+1][j+1]-2*fin_all->data[Nr+1][j]+fin_all->data[Nr+1][j-1]))/rc->data[Nr+2][0];
    fin_all->data[0][j]=(rf->data[0][0]/rf->data[1][0]*(rc->data[2][0]*fin_all->data[3][j]+(rc->data[1][0]-2*rf->data[1][0])*fin_all->data[2][j])-(rc->data[1][0]-2*rf->data[0][0])*fin_all->data[2][j]+dr*dr/ds/ds*rf->data[0][0]*(1.0/rf->data[1][0]/rf->data[1][0]-1.0/rf->data[0][0]/rf->data[0][0])*(fin_all->data[2][j+1]-2*fin_all->data[2][j]+fin_all->data[2][j-1]))/rc->data[0][0];
                                                                                                    
    }
    //利用差分表达二次拉普拉斯(去掉中心点)
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<Nr;i++)
        for(j=0;j<Ns;j++)
            fout->data[i][j]=
            (rc->data[i+2][0]*((rc->data[i+3][0]*fin_all->data[i+4][j+2]-2*rf->data[i+2][0]*fin_all->data[i+3][j+2])/rf->data[i+2][0]/dr/dr+(fin_all->data[i+3][j+3]-2*fin_all->data[i+3][j+2]+fin_all->data[i+3][j+1])/rf->data[i+2][0]/rf->data[i+2][0]/ds/ds)
            -2*rf->data[i+1][0]*((rc->data[i+2][0]*fin_all->data[i+3][j+2]+rc->data[i+1][0]*fin_all->data[i+1][j+2])/rf->data[i+1][0]/dr/dr+(fin_all->data[i+2][j+3]+fin_all->data[i+2][j+1])/rf->data[i+1][0]/rf->data[i+1][0]/ds/ds)
            +rc->data[i+1][0]*((-2*rf->data[i][0]*fin_all->data[i+1][j+2]+rc->data[i][0]*fin_all->data[i][j+2])/rf->data[i][0]/dr/dr+(fin_all->data[i+1][j+3]-2*fin_all->data[i+1][j+2]+fin_all->data[i+1][j+1])/rf->data[i][0]/rf->data[i][0]/ds/ds))*urlap->data[i][j]
            +((rc->data[i+2][0]*fin_all->data[i+3][j+3]-2*rf->data[i+1][0]*fin_all->data[i+2][j+3]+rc->data[i+1][0]*fin_all->data[i+1][j+3])/rf->data[i+1][0]/dr/dr+(fin_all->data[i+2][j+4]-2*fin_all->data[i+2][j+3])/rf->data[i+1][0]/rf->data[i+1][0]/ds/ds
            -2*((rc->data[i+2][0]*fin_all->data[i+3][j+2]+rc->data[i+1][0]*fin_all->data[i+1][j+2])/rf->data[i+1][0]/dr/dr+(fin_all->data[i+2][j+3]+fin_all->data[i+2][j+1])/rf->data[i+1][0]/rf->data[i+1][0]/ds/ds)
            +(rc->data[i+2][0]*fin_all->data[i+3][j+1]-2*rf->data[i+1][0]*fin_all->data[i+2][j+1]+rc->data[i+1][0]*fin_all->data[i+1][j+1])/rf->data[i+1][0]/dr/dr+(-2*fin_all->data[i+2][j+1]+fin_all->data[i+2][j])/rf->data[i+1][0]/rf->data[i+1][0]/ds/ds
                )*uslap->data[i][j];
    
    matrix_free(fin_all);
    
}

/*计算右端项显式部分*/
void getrhsexp(void)
{
    matrix *lapchpexp_s=0,*lapchpexp_p=0,*chpexp_s=0,*chpexp_p=0;
    matrix *phi_n=0,*chi_ns=0,*chi_np=0,*chi_sp=0;
    lapchpexp_s=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nr,Ns,lapchpexp_s);
    lapchpexp_p=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nr,Ns,lapchpexp_p);
    chpexp_s=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nr,Ns,chpexp_s);
    chpexp_p=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nr,Ns,chpexp_p);
    phi_n=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nr,Ns,phi_n);
    chi_ns=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nr,Ns,chi_ns);
    chi_np=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nr,Ns,chi_np);
    chi_sp=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nr,Ns,chi_sp);

    int i=0,j=0;
    //计算phi_n,chi_ns,chi_np,chi_sp
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<Nr;i++)
        for(j=0;j<Ns;j++)
        {
            phi_n->data[i][j]=1.0-phi_p->data[i][j]-phi_s->data[i][j];
            if(phi_n->data[i][j]<val_cut)
                phi_n->data[i][j]=val_cut;
            else if(phi_n->data[i][j]>1.0-val_cut)
                phi_n->data[i][j]=1.0-val_cut;
            
            chi_ns->data[i][j]=-0.058/(1.0-0.622*phi_s->data[i][j]*(1.0-phi_p->data[i][j]));
            chi_np->data[i][j]=3.5;
            chi_sp->data[i][j]=-1.0+0.5*phi_p->data[i][j];
        }
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<Nr;i++)
        for(j=0;j<Ns;j++)
        {
            chpexp_s->data[i][j]=log(phi_s->data[i][j])-log(phi_n->data[i][j])+chi_ns->data[i][j]*(phi_n->data[i][j]-phi_s->data[i][j])+phi_p->data[i][j]*(chi_sp->data[i][j]-chi_np->data[i][j]);
            
            chpexp_p->data[i][j]=log(phi_p->data[i][j])/mp-log(phi_n->data[i][j])+1.0/mp-1.0+chi_np->data[i][j]*(phi_n->data[i][j]-phi_p->data[i][j])+phi_s->data[i][j]*(chi_sp->data[i][j]-chi_ns->data[i][j]);
        }
    
    laplacian(chpexp_s,lapchpexp_s);
    laplacian_p(chpexp_p,lapchpexp_p);
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<Nr;i++)
        for(j=0;j<Ns;j++)
        {
            hexp_s->data[i][j]=M_ss*dt*lapchpexp_s->data[i][j];
            hexp_p->data[i][j]=dt*lapchpexp_p->data[i][j];
        }
    
    matrix_free(lapchpexp_p);
    matrix_free(lapchpexp_s);
    matrix_free(chpexp_s);
    matrix_free(chpexp_p);
    matrix_free(phi_n);
    matrix_free(chi_ns);
    matrix_free(chi_sp);
    matrix_free(chi_np);
}

/*计算右端项隐式部分*/
void getrhsimp(void)
{
    int i=0,j=0;
    matrix *doublelaplacianphi_s=0,*doublelaplacianphi_p=0;
    doublelaplacianphi_s=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nr,Ns,doublelaplacianphi_s);
    doublelaplacianphi_p=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nr,Ns,doublelaplacianphi_p);
    doublelaplacian(phi_s,doublelaplacianphi_s);
    doublelaplacian_p(phi_p,doublelaplacianphi_p);
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<Nr;i++)
           for(j=0;j<Ns;j++)
           {
               himp_s->data[i][j]=-M_ss*K_ss*dt*doublelaplacianphi_s->data[i][j];
               himp_p->data[i][j]=-K_pp*dt*doublelaplacianphi_p->data[i][j];
               
           }
    matrix_free(doublelaplacianphi_p);
    matrix_free(doublelaplacianphi_s);
}

/*rktvd算法*/
void tmintrktvd(void)
{
    int i=0,j=0;
    matrix *phi_sn=0,*phi_pn=0,*phi_s1=0,*phi_p1=0,*phi_s2=0,*phi_p2=0;
    phi_sn=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nr,Ns,phi_sn);
    phi_pn=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nr,Ns,phi_pn);
    phi_s1=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nr,Ns,phi_s1);
    phi_p1=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nr,Ns,phi_p1);
    phi_s2=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nr,Ns,phi_s2);
    phi_p2=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nr,Ns,phi_p2);
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<Nr;i++)
        for(j=0;j<Ns;j++)
        {
             phi_sn->data[i][j] = phi_s->data[i][j];
             phi_pn->data[i][j] = phi_p->data[i][j];
        }

    getrhsexp();
    getrhsimp();
    
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<Nr;i++)
        for(j=0;j<Ns;j++)
        {
            phi_s->data[i][j]=phi_sn->data[i][j]+hexp_s->data[i][j]+himp_s->data[i][j];
            phi_p->data[i][j]=phi_pn->data[i][j]+hexp_p->data[i][j]+himp_p->data[i][j];
            
            if(phi_s->data[i][j]<val_cut)
                phi_s->data[i][j]=val_cut;
            else if(phi_s->data[i][j]>1.0-val_cut)
                phi_s->data[i][j]=1.0-val_cut;
            
            if(phi_p->data[i][j]<val_cut)
                phi_p->data[i][j]=val_cut;
            else if(phi_p->data[i][j]>1.0-val_cut)
                phi_p->data[i][j]=1.0-val_cut;
        }
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<Nr;i++)
        for(j=0;j<Ns;j++)
        {
             phi_s1->data[i][j] = phi_s->data[i][j];
             phi_p1->data[i][j] = phi_p->data[i][j];
        }
    
    getrhsexp();
    getrhsimp();
    
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<Nr;i++)
        for(j=0;j<Ns;j++)
        {
            phi_s->data[i][j]=3.0/4*phi_sn->data[i][j]+1.0/4*phi_s1->data[i][j]+1.0/4*(hexp_s->data[i][j]+himp_s->data[i][j]);
            phi_p->data[i][j]=3.0/4*phi_pn->data[i][j]+1.0/4*phi_p1->data[i][j]+1.0/4*(hexp_p->data[i][j]+himp_p->data[i][j]);
            
            if(phi_s->data[i][j]<val_cut)
                phi_s->data[i][j]=val_cut;
            else if(phi_s->data[i][j]>1.0-val_cut)
                phi_s->data[i][j]=1.0-val_cut;
            
            if(phi_p->data[i][j]<val_cut)
                phi_p->data[i][j]=val_cut;
            else if(phi_p->data[i][j]>1.0-val_cut)
                phi_p->data[i][j]=1.0-val_cut;
        }
    
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<Nr;i++)
        for(j=0;j<Ns;j++)
        {
             phi_s2->data[i][j] = phi_s->data[i][j];
             phi_p2->data[i][j] = phi_p->data[i][j];
        }
       
    getrhsexp();
    getrhsimp();
    
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<Nr;i++)
        for(j=0;j<Ns;j++)
    {
        phi_s->data[i][j]=1.0/3*phi_sn->data[i][j]+2.0/3*phi_s2->data[i][j]+2.0/3*(hexp_s->data[i][j]+himp_s->data[i][j]);
        phi_p->data[i][j]=1.0/3*phi_pn->data[i][j]+2.0/3*phi_p2->data[i][j]+2.0/3*(hexp_p->data[i][j]+himp_p->data[i][j]);
        
        if(phi_s->data[i][j]<val_cut)
            phi_s->data[i][j]=val_cut;
        else if(phi_s->data[i][j]>1.0-val_cut)
            phi_s->data[i][j]=1.0-val_cut;
        
        if(phi_p->data[i][j]<val_cut)
            phi_p->data[i][j]=val_cut;
        else if(phi_p->data[i][j]>1.0-val_cut)
            phi_p->data[i][j]=1.0-val_cut;
    }
    
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<Nr;i++)
        for(j=0;j<Ns;j++)
    {
          hexp_s_np->data[i][j]=hexp_s->data[i][j];
          hexp_p_np->data[i][j]=hexp_p->data[i][j];
    }
    
    matrix_free(phi_sn);
    matrix_free(phi_pn);
    matrix_free(phi_s1);
    matrix_free(phi_p1);
    matrix_free(phi_s2);
    matrix_free(phi_p2);
}

/*Adams-Bashforth+Crank-Nicholson算法*/
void tmintabcn(void)
{
    int i=0,j=0,it=0;
    double omg=1.0,error=0;
    matrix *rhs_s=0,*rhs_p=0,*utmp=0,*uout=0,*err_s=0,*err_p=0,*doublelap_utmp=0;
    utmp=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nr,Ns,utmp);
    rhs_s=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nr,Ns,rhs_s);
    rhs_p=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nr,Ns,rhs_p);
    err_s=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nr,Ns,err_s);
    err_p=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nr,Ns,err_p);
    doublelap_utmp=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nr,Ns,doublelap_utmp);
    
    uout=(matrix*)malloc(sizeof(matrix));
    
    getrhsexp();
    getrhsimp();
    
    //phi_s
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<Nr;i++)
       for(j=0;j<Ns;j++)
    {
        rhs_s->data[i][j]=phi_s->data[i][j]+0.5*himp_s->data[i][j]+(1.5*hexp_s->data[i][j]-0.5*hexp_s_np->data[i][j]);
    }
    
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<Nr;i++)
       for(j=0;j<Ns;j++)
    {
        utmp->data[i][j]=phi_s->data[i][j];
    }
    
    matrix_zeros(Nr,Ns,uout);
    
    it=0;
    matrix_minus(uout,utmp,err_s);
    error=matrix_root_mean_square(err_s);
    while((it<100)&&(error>1e-12))
    {
        doublelaplacian_withoutcenter(utmp,doublelap_utmp);
        #pragma omp parallel for num_threads(NUMTHREAD)
        for(i=0;i<Nr;i++)
                 for(j=0;j<Ns;j++)
              {
                  uout->data[i][j]=utmp->data[i][j]*(1.0-omg)+omg*(rhs_s->data[i][j]-0.5*M_ss*K_ss*dt*doublelap_utmp->data[i][j])*(1.0/(1.0+dt*M_ss*K_ss*(coefficient_r->data[i][j]+coefficient_s->data[i][j])*0.5));
              }
        matrix_minus(uout,utmp,err_s);
        error=matrix_root_mean_square(err_s);
        for(i=0;i<Nr;i++)
           for(j=0;j<Ns;j++)
        {
            utmp->data[i][j]=uout->data[i][j];
        }
        it=it+1;
    }
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<Nr;i++)
       for(j=0;j<Ns;j++)
    {
        phi_s->data[i][j]=uout->data[i][j];
        hexp_s_np->data[i][j]=hexp_s->data[i][j];
        
        if(phi_s->data[i][j]<val_cut)
            phi_s->data[i][j]=val_cut;
        else if(phi_s->data[i][j]>1.0-val_cut)
            phi_s->data[i][j]=1.0-val_cut;
    }
    
    matrix_free(uout);
    
    
    
    //phi_p
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<Nr;i++)
       for(j=0;j<Ns;j++)
    {
        rhs_p->data[i][j]=phi_p->data[i][j]+0.5*himp_p->data[i][j]+(1.5*hexp_p->data[i][j]-0.5*hexp_p_np->data[i][j]);
    }
    
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<Nr;i++)
       for(j=0;j<Ns;j++)
    {
        utmp->data[i][j]=phi_p->data[i][j];
    }
    
    matrix_zeros(Nr,Ns,uout);
    it=0;
    matrix_minus(uout,utmp,err_p);
    error=matrix_root_mean_square(err_p);
    while((it<100)&&(error>1e-12))
    {
        doublelaplacian_withoutcenter_p(utmp,doublelap_utmp);
        #pragma omp parallel for num_threads(NUMTHREAD)
        for(i=0;i<Nr;i++)
                 for(j=0;j<Ns;j++)
              {
                  uout->data[i][j]=utmp->data[i][j]*(1.0-omg)+omg*(rhs_p->data[i][j]-0.5*K_pp*dt*doublelap_utmp->data[i][j])*(1.0/(1.0+dt*K_pp*(M_ppr*coefficient_r->data[i][j]+M_pps*coefficient_s->data[i][j])*0.5));
              }
        matrix_minus(uout,utmp,err_p);
        error=matrix_root_mean_square(err_p);
        #pragma omp parallel for num_threads(NUMTHREAD)
        for(i=0;i<Nr;i++)
           for(j=0;j<Ns;j++)
        {
            utmp->data[i][j]=uout->data[i][j];
        }
        it=it+1;
    }
    #pragma omp parallel for num_threads(NUMTHREAD)
    for(i=0;i<Nr;i++)
       for(j=0;j<Ns;j++)
    {
        phi_p->data[i][j]=uout->data[i][j];
        hexp_p_np->data[i][j]=hexp_p->data[i][j];
 
        if(phi_p->data[i][j]<val_cut)
            phi_p->data[i][j]=val_cut;
        else if(phi_p->data[i][j]>1.0-val_cut)
            phi_p->data[i][j]=1.0-val_cut;
    }
    
    matrix_free(uout);
    
    matrix_free(rhs_s);
    matrix_free(rhs_p);
    matrix_free(utmp);
    matrix_free(err_s);
    matrix_free(err_p);
    matrix_free(doublelap_utmp);
    
}


/*初始化*/
void initialization(void)
{
    matrix *Gaussian_blur_p=0,*Gaussian_blur_s=0;
    Gaussian_blur_p=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nr,Ns,Gaussian_blur_p);
    Gaussian_blur_s=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nr,Ns,Gaussian_blur_s);
    int i=0,j=0,k=0;
    //赋初值
    for(i=0;i<Nr;i++)
    for(j=0;j<Ns;j++)
    {
        if(i*dr+Ri<Ifi||i*dr+Ri>Ifo)
        {
            phi_s->data[i][j]=0.01;
            phi_p->data[i][j]=0.01;
        }
        else
        {
            phi_s->data[i][j]=0.65;
            phi_p->data[i][j]=0.30;
        }
        
    }
    //高斯光滑(3次)
    for(k=1;k<=3;k++)
    {
        for(i=0;i<Nr;i++)
            for(j=0;j<Ns;j++)
            {
                Gaussian_blur_p->data[i][j]=phi_p->data[i][j];
                Gaussian_blur_s->data[i][j]=phi_s->data[i][j];
            }
    for(j=0;j<=Ns-1;j++)
        for(i=2;i<=Nr-3;i++)
        {
            phi_s->data[i][j]=1.105682211*(0.110865*(Gaussian_blur_s->data[i-2][j]+Gaussian_blur_s->data[i+2][j])+0.210786*(Gaussian_blur_s->data[i-1][j]+Gaussian_blur_s->data[i+1][j])+0.261117*Gaussian_blur_s->data[i][j]);
            phi_p->data[i][j]=1.105682211*(0.110865*(Gaussian_blur_p->data[i-2][j]+Gaussian_blur_p->data[i+2][j])+0.210786*(Gaussian_blur_p->data[i-1][j]+Gaussian_blur_p->data[i+1][j])+0.261117*Gaussian_blur_p->data[i][j]);
        }
    }
    
    //加扰动
    double eps=1e-3;
    double noise=0;
    srand(time(NULL));
    for(i=0;i<Nr;i++)
    for(j=0;j<Ns;j++)
    {
        if(i*dr+Ri>Ifi&&i*dr+Ri<Ifo)
        {
            noise=eps*(2*((double)rand()/RAND_MAX)-1.0);
            phi_s->data[i][j]=phi_s->data[i][j]-noise;
            phi_p->data[i][j]=phi_p->data[i][j]+noise;
            
            if(phi_s->data[i][j]<val_cut)
            phi_s->data[i][j]=val_cut;
            else if(phi_s->data[i][j]>1.0-val_cut)
            phi_s->data[i][j]=1.0-val_cut;
            
            if(phi_p->data[i][j]<val_cut)
            phi_p->data[i][j]=val_cut;
            else if(phi_p->data[i][j]>1.0-val_cut)
            phi_p->data[i][j]=1.0-val_cut;
        }
    }
    
}

int main(void)
{
    int i=0,j=0;
    FILE *fp1,*fp2,*fp3,*fp4;
    fp1=fopen("phis0.txt","r");
    fp2=fopen("phip0.txt","r");
    fp3=fopen("phis.txt","w");
    fp4=fopen("phip.txt","w");
    // parameters
    mp=5.0;
    
    M_ppr=2;
    M_pps=4;
    M_ss=2000;
    
    M_sp=0.0;
    M_ps=0.0;
    
    K_pp=1e-4;
    K_ss=1e-4;
    
    Ri=0.35;
    Ifi=0.70;
    Ifo=1.1;
    Ro=1.45;
    
    Lr=Ro-Ri;
    Ls=atan(1.0)*4.0/6.0;
    
    Nr = 200;
    Ns = 100;
    
    dr = Lr/Nr;
    ds = Ls/Ns;
    
    dt = 1e-11;
    
    val_cut = 1e-6;
    
    rc=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nr+3,1,rc);
    rf=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nr+2,1,rf);
    
    for(i=0;i<=Nr+2;i++)
    {
        rc->data[i][0]=(0.5+(double)(i-2))*dr+Ri;
    }
    for(i=0;i<=Nr+1;i++)
    {
        rf->data[i][0]=(double)(i-1)*dr+Ri;
    }
    
    uslap=(matrix*)malloc(sizeof(matrix));
    urlap=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nr,Ns,uslap);
    matrix_create(Nr,Ns,urlap);
    
    for(i=0;i<Nr;i++)
    {
        for(j=0;j<Ns;j++)
        {
            urlap->data[i][j]=1.0/dr/dr/rf->data[i+1][0];
            uslap->data[i][j]=1.0/ds/ds/rf->data[i+1][0]/rf->data[i+1][0];
        }
    }
    
    
    // field vars
    phi_s=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nr,Ns,phi_s);
    phi_p=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nr,Ns,phi_p);
    
    chp_p=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nr,Ns,chp_p);
    chp_s=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nr,Ns,chp_s);
    
    dPsi_ds=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nr,Ns,dPsi_ds);
    dPsi_dp=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nr,Ns,dPsi_dp);
    
    coefficient_r=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nr,Ns,coefficient_r);
    coefficient_s=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nr,Ns,coefficient_s);
    for(i=0;i<Nr;i++)
    {
        for(j=0;j<Ns;j++)
        {
            coefficient_r->data[i][j]=
            rc->data[i+2][0]*rc->data[i+2][0]/rf->data[i+1][0]/rf->data[i+2][0]/dr/dr/dr/dr+4.0/dr/dr/dr/dr+4.0/rf->data[i+1][0]/rf->data[i+1][0]/dr/dr/ds/ds+rc->data[i+1][0]*rc->data[i+1][0]/rf->data[i+1][0]/rf->data[i][0]/dr/dr/dr/dr;
            
            coefficient_s->data[i][j]=
            +1.0/rf->data[i+1][0]/rf->data[i+1][0]/rf->data[i+1][0]/rf->data[i+1][0]/ds/ds/ds/ds+4.0/rf->data[i+1][0]/rf->data[i+1][0]/dr/dr/ds/ds+4.0/rf->data[i+1][0]/rf->data[i+1][0]/rf->data[i+1][0]/rf->data[i+1][0]/ds/ds/ds/ds+1.0/rf->data[i+1][0]/rf->data[i+1][0]/rf->data[i+1][0]/rf->data[i+1][0]/ds/ds/ds/ds;
        }
    }
    
     // initial fields
    //initialization();
    for(i=0;i<Nr;i++)
    {
        for(j=0;j<Ns;j++)
        {
            fscanf(fp1,"%lf ",&phi_s->data[i][j]);
            fscanf(fp2,"%lf ",&phi_p->data[i][j]);

        }
        fscanf(fp1,"\n");
        fscanf(fp2,"\n");
    }
    fclose(fp1);
    fclose(fp2);
    
    //时间积分
    himp_s=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nr,Ns,himp_s);
    himp_p=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nr,Ns,himp_p);
    
    hexp_s=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nr,Ns,hexp_s);
    hexp_p=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nr,Ns,hexp_p);
    hexp_s_np=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nr,Ns,hexp_s_np);
    hexp_p_np=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nr,Ns,hexp_p_np);
    
    double error=0;
    error=matrix_max(phi_p);
    int it=1;
    
    matrix *phi_p_np, *phi_s_np,*error_s,*error_p;
    phi_p_np=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nr,Ns,phi_p_np);
    phi_s_np=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nr,Ns,phi_s_np);
    error_s=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nr,Ns,error_s);
    error_p=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nr,Ns,error_p);
    while(it<=5000000)
    {
        for(i=0;i<Nr;i++)
            for(j=0;j<Ns;j++)
            {
                  phi_s_np->data[i][j]=phi_s->data[i][j];
                  phi_p_np->data[i][j]=phi_p->data[i][j];
            }

        if(it==1)
            tmintrktvd();         /*时间rktvd积分*/
        else
            tmintabcn();         /*时间AB-CN积分*/
        
        matrix_minus(phi_s,phi_s_np,error_s);
        matrix_minus(phi_p,phi_p_np,error_p);
        
        if(it%2000==0)
        {
            printf("Step: %d;  Time: %e;  error phi_p: %e;  error phi_s: %e\n",it, dt*it, matrix_standarderror(error_p), matrix_standarderror(error_s) );
        }
        error=matrix_standarderror(error_p);
        it=it+1;
    }
    
     for(i=0;i<Nr;i++)
    {
        for(j=0;j<Ns;j++)
        {
            fprintf(fp3,"%e ",phi_s->data[i][j]);
            fprintf(fp4,"%e ",phi_p->data[i][j]);
        }
        fprintf(fp3,"\n");
        fprintf(fp4,"\n");
    }
    fclose(fp3);
    fclose(fp4);
    
    matrix_free(phi_p_np);
    matrix_free(phi_s_np);
    matrix_free(error_p);
    matrix_free(error_s);
    
    
    return 0;
}
