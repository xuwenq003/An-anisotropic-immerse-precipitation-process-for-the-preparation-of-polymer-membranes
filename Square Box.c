//
//  main.c
//  phase field simulations
//
//  Created by Mac on 2019/1/23.
//  Copyright © 2019 Mac. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct
{
    int row;   //矩阵行数
    int col;   //矩阵列数
    double ** data;   //二维数组作为矩阵
}
matrix;

//变量
/*physical properties*/
double M_sp,M_ps,K_pp,K_ss,mp;
double M_ppx,M_ppy,M_ssx,M_ssy;

/*geometry*/
double Lx,Ly,Ifi,Ifo,dx,dy,dt,val_cut;
int Nx,Ny;

/*grids*/
matrix *xc,*yc,*xc1d,*yc1d;

/*concentration*/
matrix *phi_s,*phi_p;

/*other field*/
matrix *chp_p,*chp_s,*dPsi_ds,*dPsi_dp,*hexp_s,*hexp_p,*himp_s,*himp_p;
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
    for(i=0;i<row;i++)
        for(j=0;j<col;j++)
            mat->data[i][j]=0.0;
}

/*生成全“1”矩阵*/
void matrix_ones(int row,int col,matrix *mat)
{
    int i=0,j=0;
    matrix_create(row,col,mat);
    for(i=0;i<row;i++)
        for(j=0;j<col;j++)
            mat->data[i][j]=1.0;
}

/*删除矩阵*/
void matrix_free(matrix *mat)
{
    int i=0;
    for (i = 0;i < mat->row;i++)
    {
        free (mat->data[i]);
    }
    free (mat->data);
}

/*矩阵相加*/
void matrix_add(matrix *mat1,matrix *mat2,matrix *matresult)
{
    int i=0,j=0;
    for(i=0;i<matresult->row;i++)
        for(j=0;j<matresult->col;j++)
            matresult->data[i][j]=mat1->data[i][j]+mat2->data[i][j];
}

/*矩阵相减*/
void matrix_minus(matrix *mat1,matrix *mat2,matrix *matresult)
{
    int i=0,j=0;
    for(i=0;i<matresult->row;i++)
        for(j=0;j<matresult->col;j++)
            matresult->data[i][j]=mat1->data[i][j]-mat2->data[i][j];
}

/*矩阵数乘*/
void matrix_multiply(double coefficient,matrix *mat,matrix *matresult)
{
    int i=0,j=0;
    for(i=0;i<mat->row;i++)
        for(j=0;j<mat->col;j++)
            matresult->data[i][j]=mat->data[i][j]*coefficient;
}

/*矩阵乘法（点乘）*/
void matrix_dotproduct(matrix *mat1,matrix *mat2,matrix *matresult)
{
    int i=0,j=0,k=0;
    for(i=0;i<mat1->row;i++)
        for(j=0;j<mat2->col;j++)
            for(k=0;k<mat1->col;k++)
                matresult->data[i][j]+=mat1->data[i][k]+mat2->data[k][j];
}


/*矩阵转置*/
void matrix_transpose(matrix *mat,matrix *mat_t)
{
    int i=0,j=0;
    for(i=0;i<mat->row;i++)
        for(j=0;j<mat->col;j++)
            mat_t->data[j][i]=mat->data[i][j];
}

/*矩阵平铺*/
void matrix_replication(matrix *mat,matrix *matresult,int rowrep,int colrep)
{
    int i=0,j=0;
    for(i=0;i<mat->row*rowrep;i++)
        for(j=0;j<mat->col*colrep;j++)
            matresult->data[i][j]=mat->data[i%mat->row][j%mat->col];
}

/*矩阵打印*/
void matrix_print(matrix *mat)
{
    int i=0,j=0;
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
        {
            root_mean_square+=(mat->data[i][j]*mat->data[i][j]);
        }
    root_mean_square=sqrt(root_mean_square/(mat->col*mat->row));
    return root_mean_square;
}



/*拉普拉斯算子s*/
void laplacian_s(matrix *fin,matrix *fout)
{
    int i=0,j=0;
    matrix *fin_all=0;
    fin_all=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nx+2,Ny+2,fin_all);
    //中间部分
    for(i=0;i<Nx;i++)
    {
        for(j=0;j<Ny;j++)
        {
            fin_all->data[i+1][j+1]=fin->data[i][j];
        }
    }
    //边界
    for(j=1;j<=Ny;j++)
    {
        fin_all->data[0][j]=fin_all->data[Nx][j];
        fin_all->data[Nx+1][j]=fin_all->data[1][j];
    }
    for(i=0;i<=Nx+1;i++)
    {
        fin_all->data[i][Ny+1]=fin_all->data[i][1];
        fin_all->data[i][0]=fin_all->data[i][Ny];
    }

    //利用差分表达拉普拉斯
    for(i=0;i<Nx;i++)
        for(j=0;j<Ny;j++)
            fout->data[i][j]=
            M_ssx*(fin_all->data[i+2][j+1]-2.0*fin_all->data[i+1][j+1]+fin_all->data[i][j+1])/dx/dx            +M_ssy*(fin_all->data[i+1][j+2]-2.0*fin_all->data[i+1][j+1]+fin_all->data[i+1][j])/dy/dy;
    
    matrix_free(fin_all);
}

/*拉普拉斯算子p*/
void laplacian_p(matrix *fin,matrix *fout)
{
    int i=0,j=0;
    matrix *fin_all=0;
    fin_all=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nx+2,Ny+2,fin_all);
    //中间部分
    for(i=0;i<Nx;i++)
    {
        for(j=0;j<Ny;j++)
        {
            fin_all->data[i+1][j+1]=fin->data[i][j];
        }
    }
    //边界
    for(j=1;j<=Ny;j++)
    {
        fin_all->data[0][j]=fin_all->data[Nx][j];
        fin_all->data[Nx+1][j]=fin_all->data[1][j];
    }
    for(i=0;i<=Nx+1;i++)
    {
        fin_all->data[i][Ny+1]=fin_all->data[i][1];
        fin_all->data[i][0]=fin_all->data[i][Ny];
    }

    //利用差分表达拉普拉斯
    for(i=0;i<Nx;i++)
        for(j=0;j<Ny;j++)
            fout->data[i][j]=
            M_ppx*(fin_all->data[i+2][j+1]-2.0*fin_all->data[i+1][j+1]+fin_all->data[i][j+1])/dx/dx            +M_ppy*(fin_all->data[i+1][j+2]-2.0*fin_all->data[i+1][j+1]+fin_all->data[i+1][j])/dy/dy;
    
    matrix_free(fin_all);
}

/*二次拉普拉斯算子s*/
void doublelaplacian_s(matrix *fin,matrix *fout)
{
    int i=0,j=0;
    matrix *fin_all=0;
    fin_all=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nx+4,Ny+4,fin_all);
    //中间部分
    for(i=0;i<Nx;i++)
    {
        for(j=0;j<Ny;j++)
        {
            fin_all->data[i+2][j+2]=fin->data[i][j];
        }
    }
    //边界
    for(j=2;j<=Ny+1;j++)
    {
        fin_all->data[0][j]=fin_all->data[Nx][j];
        fin_all->data[1][j]=fin_all->data[Nx+1][j];
        fin_all->data[Nx+2][j]=fin_all->data[2][j];
        fin_all->data[Nx+3][j]=fin_all->data[3][j];
    }
    for(i=0;i<=Nx+3;i++)
    {
        fin_all->data[i][Ny+2]=fin_all->data[i][2];
        fin_all->data[i][1]=fin_all->data[i][Ny+1];
        fin_all->data[i][Ny+3]=fin_all->data[i][3];
        fin_all->data[i][0]=fin_all->data[i][Ny];
    }
    //利用差分表达二次拉普拉斯
    for(i=0;i<Nx;i++)
        for(j=0;j<Ny;j++)
        {
            fout->data[i][j]=M_ssx*(
                             (fin_all->data[i+4][j+2]-2*fin_all->data[i+3][j+2]+fin_all->data[i+2][j+2])/dx/dx+(fin_all->data[i+3][j+3]-2*fin_all->data[i+3][j+2]+fin_all->data[i+3][j+1])/dy/dy
                             -2*(
                                 (fin_all->data[i+3][j+2]-2*fin_all->data[i+2][j+2]+fin_all->data[i+1][j+2])/dx/dx+(fin_all->data[i+2][j+3]-2*fin_all->data[i+2][j+2]+fin_all->data[i+2][j+1])/dy/dy
                                 )
                             +
                             (fin_all->data[i+2][j+2]-2*fin_all->data[i+1][j+2]+fin_all->data[i][j+2])/dx/dx+(fin_all->data[i+1][j+3]-2*fin_all->data[i+1][j+2]+fin_all->data[i+1][j+1])/dy/dy
                             )/dx/dx
            +M_ssy*(
              (fin_all->data[i+3][j+3]-2*fin_all->data[i+2][j+3]+fin_all->data[i+1][j+3])/dx/dx+(fin_all->data[i+2][j+4]-2*fin_all->data[i+2][j+3]+fin_all->data[i+2][j+2])/dy/dy
              -2*(
                  (fin_all->data[i+3][j+2]-2*fin_all->data[i+2][j+2]+fin_all->data[i+1][j+2])/dx/dx+(fin_all->data[i+2][j+3]-2*fin_all->data[i+2][j+2]+fin_all->data[i+2][j+1])/dy/dy
                  )
              +
              (fin_all->data[i+3][j+1]-2*fin_all->data[i+2][j+1]+fin_all->data[i+1][j+1])/dx/dx+(fin_all->data[i+2][j+2]-2*fin_all->data[i+2][j+1]+fin_all->data[i+2][j])/dy/dy
              )/dy/dy;
        }
    matrix_free(fin_all);
}

/*二次拉普拉斯算子p*/
void doublelaplacian_p(matrix *fin,matrix *fout)
{
    int i=0,j=0;
    matrix *fin_all=0;
    fin_all=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nx+4,Ny+4,fin_all);
    //中间部分
    for(i=0;i<Nx;i++)
    {
        for(j=0;j<Ny;j++)
        {
            fin_all->data[i+2][j+2]=fin->data[i][j];
        }
    }
    //边界
    for(j=2;j<=Ny+1;j++)
    {
        fin_all->data[0][j]=fin_all->data[Nx][j];
        fin_all->data[1][j]=fin_all->data[Nx+1][j];
        fin_all->data[Nx+2][j]=fin_all->data[2][j];
        fin_all->data[Nx+3][j]=fin_all->data[3][j];
    }
    for(i=0;i<=Nx+3;i++)
    {
        fin_all->data[i][Ny+2]=fin_all->data[i][2];
        fin_all->data[i][1]=fin_all->data[i][Ny+1];
        fin_all->data[i][Ny+3]=fin_all->data[i][3];
        fin_all->data[i][0]=fin_all->data[i][Ny];
    }
    //利用差分表达二次拉普拉斯
    for(i=0;i<Nx;i++)
        for(j=0;j<Ny;j++)
        {
            fout->data[i][j]=M_ppx*(
                             (fin_all->data[i+4][j+2]-2*fin_all->data[i+3][j+2]+fin_all->data[i+2][j+2])/dx/dx+(fin_all->data[i+3][j+3]-2*fin_all->data[i+3][j+2]+fin_all->data[i+3][j+1])/dy/dy
                             -2*(
                                 (fin_all->data[i+3][j+2]-2*fin_all->data[i+2][j+2]+fin_all->data[i+1][j+2])/dx/dx+(fin_all->data[i+2][j+3]-2*fin_all->data[i+2][j+2]+fin_all->data[i+2][j+1])/dy/dy
                                 )
                             +
                             (fin_all->data[i+2][j+2]-2*fin_all->data[i+1][j+2]+fin_all->data[i][j+2])/dx/dx+(fin_all->data[i+1][j+3]-2*fin_all->data[i+1][j+2]+fin_all->data[i+1][j+1])/dy/dy
                             )/dx/dx
            +M_ppy*(
              (fin_all->data[i+3][j+3]-2*fin_all->data[i+2][j+3]+fin_all->data[i+1][j+3])/dx/dx+(fin_all->data[i+2][j+4]-2*fin_all->data[i+2][j+3]+fin_all->data[i+2][j+2])/dy/dy
              -2*(
                  (fin_all->data[i+3][j+2]-2*fin_all->data[i+2][j+2]+fin_all->data[i+1][j+2])/dx/dx+(fin_all->data[i+2][j+3]-2*fin_all->data[i+2][j+2]+fin_all->data[i+2][j+1])/dy/dy
                  )
              +
              (fin_all->data[i+3][j+1]-2*fin_all->data[i+2][j+1]+fin_all->data[i+1][j+1])/dx/dx+(fin_all->data[i+2][j+2]-2*fin_all->data[i+2][j+1]+fin_all->data[i+2][j])/dy/dy
              )/dy/dy;
        }
    matrix_free(fin_all);
}

/*去中心的二次拉普拉斯算子s*/
void doublelaplacian_withoutcenter_s(matrix* fin,matrix* fout)
{
    int i=0,j=0;
    matrix *fin_all=0;
    fin_all=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nx+4,Ny+4,fin_all);
    //中间部分
    for(i=0;i<Nx;i++)
    {
        for(j=0;j<Ny;j++)
        {
            fin_all->data[i+2][j+2]=fin->data[i][j];
        }
    }
    //边界
    for(j=2;j<=Ny+1;j++)
    {
        fin_all->data[0][j]=fin_all->data[Nx][j];
        fin_all->data[1][j]=fin_all->data[Nx+1][j];
        fin_all->data[Nx+2][j]=fin_all->data[2][j];
        fin_all->data[Nx+3][j]=fin_all->data[3][j];
    }
    for(i=0;i<=Nx+3;i++)
    {
        fin_all->data[i][Ny+2]=fin_all->data[i][2];
        fin_all->data[i][1]=fin_all->data[i][Ny+1];
        fin_all->data[i][Ny+3]=fin_all->data[i][3];
        fin_all->data[i][0]=fin_all->data[i][Ny];
    }
    //利用差分表达二次拉普拉斯(去掉中心点)
    for(i=0;i<Nx;i++)
        for(j=0;j<Ny;j++)
            fout->data[i][j]=M_ssx*(
                             (fin_all->data[i+4][j+2]-2*fin_all->data[i+3][j+2])/dx/dx+(fin_all->data[i+3][j+3]-2*fin_all->data[i+3][j+2]+fin_all->data[i+3][j+1])/dy/dy
                             -2*(
                                 (fin_all->data[i+3][j+2]+fin_all->data[i+1][j+2])/dx/dx+(fin_all->data[i+2][j+3]+fin_all->data[i+2][j+1])/dy/dy
                                 )
                             +
                             (-2*fin_all->data[i+1][j+2]+fin_all->data[i][j+2])/dx/dx+(fin_all->data[i+1][j+3]-2*fin_all->data[i+1][j+2]+fin_all->data[i+1][j+1])/dy/dy
                             )/dx/dx
            +M_ssy*(
              (fin_all->data[i+3][j+3]-2*fin_all->data[i+2][j+3]+fin_all->data[i+1][j+3])/dx/dx+(fin_all->data[i+2][j+4]-2*fin_all->data[i+2][j+3])/dy/dy
              -2*(
                  (fin_all->data[i+3][j+2]+fin_all->data[i+1][j+2])/dx/dx+(fin_all->data[i+2][j+3]+fin_all->data[i+2][j+1])/dy/dy
                  )
              +
              (fin_all->data[i+3][j+1]-2*fin_all->data[i+2][j+1]+fin_all->data[i+1][j+1])/dx/dx+(-2*fin_all->data[i+2][j+1]+fin_all->data[i+2][j])/dy/dy
              )/dy/dy;
    matrix_free(fin_all);
}

/*去中心的二次拉普拉斯算子p*/
void doublelaplacian_withoutcenter_p(matrix* fin,matrix* fout)
{
    int i=0,j=0;
    matrix *fin_all=0;
    fin_all=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nx+4,Ny+4,fin_all);
    //中间部分
    for(i=0;i<Nx;i++)
    {
        for(j=0;j<Ny;j++)
        {
            fin_all->data[i+2][j+2]=fin->data[i][j];
        }
    }
    //边界
    for(j=2;j<=Ny+1;j++)
    {
        fin_all->data[0][j]=fin_all->data[Nx][j];
        fin_all->data[1][j]=fin_all->data[Nx+1][j];
        fin_all->data[Nx+2][j]=fin_all->data[2][j];
        fin_all->data[Nx+3][j]=fin_all->data[3][j];
    }
    for(i=0;i<=Nx+3;i++)
    {
        fin_all->data[i][Ny+2]=fin_all->data[i][2];
        fin_all->data[i][1]=fin_all->data[i][Ny+1];
        fin_all->data[i][Ny+3]=fin_all->data[i][3];
        fin_all->data[i][0]=fin_all->data[i][Ny];
    }
    //利用差分表达二次拉普拉斯(去掉中心点)
    for(i=0;i<Nx;i++)
        for(j=0;j<Ny;j++)
            fout->data[i][j]=M_ppx*(
                             (fin_all->data[i+4][j+2]-2*fin_all->data[i+3][j+2])/dx/dx+(fin_all->data[i+3][j+3]-2*fin_all->data[i+3][j+2]+fin_all->data[i+3][j+1])/dy/dy
                             -2*(
                                 (fin_all->data[i+3][j+2]+fin_all->data[i+1][j+2])/dx/dx+(fin_all->data[i+2][j+3]+fin_all->data[i+2][j+1])/dy/dy
                                 )
                             +
                             (-2*fin_all->data[i+1][j+2]+fin_all->data[i][j+2])/dx/dx+(fin_all->data[i+1][j+3]-2*fin_all->data[i+1][j+2]+fin_all->data[i+1][j+1])/dy/dy
                             )/dx/dx
            +M_ppy*(
              (fin_all->data[i+3][j+3]-2*fin_all->data[i+2][j+3]+fin_all->data[i+1][j+3])/dx/dx+(fin_all->data[i+2][j+4]-2*fin_all->data[i+2][j+3])/dy/dy
              -2*(
                  (fin_all->data[i+3][j+2]+fin_all->data[i+1][j+2])/dx/dx+(fin_all->data[i+2][j+3]+fin_all->data[i+2][j+1])/dy/dy
                  )
              +
              (fin_all->data[i+3][j+1]-2*fin_all->data[i+2][j+1]+fin_all->data[i+1][j+1])/dx/dx+(-2*fin_all->data[i+2][j+1]+fin_all->data[i+2][j])/dy/dy
              )/dy/dy;
    matrix_free(fin_all);
}

/*计算右端项显式部分*/
void getrhsexp(void)
{
    matrix *lapchpexp_s=0,*lapchpexp_p=0,*chpexp_s=0,*chpexp_p=0;
    matrix *phi_n=0,*chi_ns=0,*chi_np=0,*chi_sp=0;
    lapchpexp_s=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nx,Ny,lapchpexp_s);
    lapchpexp_p=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nx,Ny,lapchpexp_p);
    chpexp_s=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nx,Ny,chpexp_s);
    chpexp_p=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nx,Ny,chpexp_p);
    phi_n=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nx,Ny,phi_n);
    chi_ns=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nx,Ny,chi_ns);
    chi_np=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nx,Ny,chi_np);
    chi_sp=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nx,Ny,chi_sp);

    int i=0,j=0;
    //计算phi_n,chi_ns,chi_np,chi_sp
    for(i=0;i<Nx;i++)
        for(j=0;j<Ny;j++)
        {
            phi_n->data[i][j]=1.0-phi_p->data[i][j]-phi_s->data[i][j];
            if(phi_n->data[i][j]<val_cut)
                phi_n->data[i][j]=val_cut;
            else if(phi_n->data[i][j]>1.0-val_cut)
                phi_n->data[i][j]=1.0-val_cut;
            
            chi_ns->data[i][j]=0.2;
            chi_np->data[i][j]=1;
            chi_sp->data[i][j]=0.3;
        }
    for(i=0;i<Nx;i++)
        for(j=0;j<Ny;j++)
        {
            chpexp_s->data[i][j]=log(phi_s->data[i][j])-log(phi_n->data[i][j])+chi_ns->data[i][j]*(phi_n->data[i][j]-phi_s->data[i][j])+phi_p->data[i][j]*(chi_sp->data[i][j]-chi_np->data[i][j]);
            
            chpexp_p->data[i][j]=log(phi_p->data[i][j])/mp-log(phi_n->data[i][j])+1.0/mp-1.0+chi_np->data[i][j]*(phi_n->data[i][j]-phi_p->data[i][j])+phi_s->data[i][j]*(chi_sp->data[i][j]-chi_ns->data[i][j]);
        }
    
    laplacian_s(chpexp_s,lapchpexp_s);
    laplacian_p(chpexp_p,lapchpexp_p);
    for(i=0;i<Nx;i++)
        for(j=0;j<Ny;j++)
        {
            hexp_s->data[i][j]=dt*lapchpexp_s->data[i][j]+M_sp*dt*lapchpexp_p->data[i][j];
            hexp_p->data[i][j]=M_ps*dt*lapchpexp_s->data[i][j]+dt*lapchpexp_p->data[i][j];
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
    matrix_create(Nx,Ny,doublelaplacianphi_s);
    doublelaplacianphi_p=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nx,Ny,doublelaplacianphi_p);
    doublelaplacian_s(phi_s,doublelaplacianphi_s);
    doublelaplacian_p(phi_p,doublelaplacianphi_p);
    for(i=0;i<Nx;i++)
           for(j=0;j<Ny;j++)
           {
               himp_s->data[i][j]=-K_ss*dt*doublelaplacianphi_s->data[i][j]-M_sp*K_pp*dt*doublelaplacianphi_p->data[i][j];
               himp_p->data[i][j]=-M_ps*K_ss*dt*doublelaplacianphi_s->data[i][j]-K_pp*dt*doublelaplacianphi_p->data[i][j];
               
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
    matrix_create(Nx,Ny,phi_sn);
    phi_pn=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nx,Ny,phi_pn);
    phi_s1=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nx,Ny,phi_s1);
    phi_p1=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nx,Ny,phi_p1);
    phi_s2=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nx,Ny,phi_s2);
    phi_p2=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nx,Ny,phi_p2);
    for(i=0;i<Nx;i++)
        for(j=0;j<Ny;j++)
        {
             phi_sn->data[i][j] = phi_s->data[i][j];
             phi_pn->data[i][j] = phi_p->data[i][j];
        }

    getrhsexp();
    getrhsimp();
    
    for(i=0;i<Nx;i++)
        for(j=0;j<Ny;j++)
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
    for(i=0;i<Nx;i++)
        for(j=0;j<Ny;j++)
        {
             phi_s1->data[i][j] = phi_s->data[i][j];
             phi_p1->data[i][j] = phi_p->data[i][j];
        }
    
    getrhsexp();
    getrhsimp();
    
    for(i=0;i<Nx;i++)
        for(j=0;j<Ny;j++)
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
    
    for(i=0;i<Nx;i++)
        for(j=0;j<Ny;j++)
        {
             phi_s2->data[i][j] = phi_s->data[i][j];
             phi_p2->data[i][j] = phi_p->data[i][j];
        }
       
    getrhsexp();
    getrhsimp();
    
    for(i=0;i<Nx;i++)
        for(j=0;j<Ny;j++)
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
    
    for(i=0;i<Nx;i++)
        for(j=0;j<Ny;j++)
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
    double omg=1.0,coef=0,error=0;
    matrix *rhs_s=0,*rhs_p=0,*utmp=0,*uout=0,*err_s=0,*err_p=0,*doublelap_utmp=0;
    utmp=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nx,Ny,utmp);
    rhs_s=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nx,Ny,rhs_s);
    rhs_p=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nx,Ny,rhs_p);
    err_s=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nx,Ny,err_s);
    err_p=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nx,Ny,err_p);
    doublelap_utmp=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nx,Ny,doublelap_utmp);
    
    uout=(matrix*)malloc(sizeof(matrix));
    
    getrhsexp();
    getrhsimp();
    
    //phi_s
    for(i=0;i<Nx;i++)
       for(j=0;j<Ny;j++)
    {
        rhs_s->data[i][j]=phi_s->data[i][j]+0.5*himp_s->data[i][j]+(1.5*hexp_s->data[i][j]-0.5*hexp_s_np->data[i][j]);
    }
    
    coef=1.0/(1.0+dt*K_ss*(M_ssx*(6.0/dx/dx/dx/dx+4.0/dx/dx/dy/dy)+M_ssy*(6.0/dy/dy/dy/dy+4.0/dx/dx/dy/dy))*0.5);
    for(i=0;i<Nx;i++)
       for(j=0;j<Ny;j++)
    {
        utmp->data[i][j]=phi_s->data[i][j];
    }
    
    matrix_zeros(Nx,Ny,uout);
    
    it=0;
    matrix_minus(uout,utmp,err_s);
    error=matrix_root_mean_square(err_s);
    while((it<100)&&(error>1e-12))
    {
        doublelaplacian_withoutcenter_s(utmp,doublelap_utmp);
        for(i=0;i<Nx;i++)
                 for(j=0;j<Ny;j++)
              {
                  uout->data[i][j]=utmp->data[i][j]*(1.0-omg)+omg*(rhs_s->data[i][j]-0.5*K_ss*dt*doublelap_utmp->data[i][j])*coef;
              }
        matrix_minus(uout,utmp,err_s);
        error=matrix_root_mean_square(err_s);
        for(i=0;i<Nx;i++)
           for(j=0;j<Ny;j++)
        {
            utmp->data[i][j]=uout->data[i][j];
        }
        it=it+1;
    }
    for(i=0;i<Nx;i++)
       for(j=0;j<Ny;j++)
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
    for(i=0;i<Nx;i++)
       for(j=0;j<Ny;j++)
    {
        rhs_p->data[i][j]=phi_p->data[i][j]+0.5*himp_p->data[i][j]+(1.5*hexp_p->data[i][j]-0.5*hexp_p_np->data[i][j]);
    }
    
    coef=1.0/(1.0+dt*K_pp*(M_ppx*(6.0/dx/dx/dx/dx+4.0/dx/dx/dy/dy)+M_ppy*(6.0/dy/dy/dy/dy+4.0/dx/dx/dy/dy))*0.5);
    for(i=0;i<Nx;i++)
       for(j=0;j<Ny;j++)
    {
        utmp->data[i][j]=phi_p->data[i][j];
    }
    
    matrix_zeros(Nx,Ny,uout);
    it=0;
    matrix_minus(uout,utmp,err_p);
    error=matrix_root_mean_square(err_p);
    while((it<100)&&(error>1e-12))
    {
        doublelaplacian_withoutcenter_p(utmp,doublelap_utmp);
        for(i=0;i<Nx;i++)
                 for(j=0;j<Ny;j++)
              {
                  uout->data[i][j]=utmp->data[i][j]*(1.0-omg)+omg*(rhs_p->data[i][j]-0.5*K_pp*dt*doublelap_utmp->data[i][j])*coef;
              }
        matrix_minus(uout,utmp,err_p);
        error=matrix_root_mean_square(err_p);
        for(i=0;i<Nx;i++)
           for(j=0;j<Ny;j++)
        {
            utmp->data[i][j]=uout->data[i][j];
        }
        it=it+1;
    }
    for(i=0;i<Nx;i++)
       for(j=0;j<Ny;j++)
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
    int i=0,j=0,k=0;
    matrix *Gaussian_blur_p=0,*Gaussian_blur_s=0;
    Gaussian_blur_p=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nx,Ny,Gaussian_blur_p);
    Gaussian_blur_s=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nx,Ny,Gaussian_blur_s);

    double Ifm=(Ifi+Ifo)/2;
    for(i=0;i<Nx;i++)
        for(j=0;j<Ny;j++)
        {
             
            phi_s->data[i][j]=0.20;
            phi_p->data[i][j]=0.20;
            
        }
    
    //加扰动
    double eps=1e-3;
    double noise=0;
    srand(time(NULL));
    for(i=0;i<Nx;i++)
        for(j=0;j<Ny;j++)
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
    matrix_free(Gaussian_blur_p);
    matrix_free(Gaussian_blur_s);
    
}


int main(void)
{
    int i=0,j=0;
    FILE *fp1,*fp2;
    fp1=fopen("phis.txt","w");
    fp2=fopen("phip.txt","w");
    // parameters
    mp=64.0;
    
    M_ppx=2;
    M_ppy=2;
    M_ssx=2;
    M_ssy=2;
    
    M_sp=0.0;
    M_ps=0.0;
    
    K_pp=1.6e-5;
    K_ss=1.6e-5;
    
    Lx = 1.0;
    Ly = 1.0;
    Ifi=0.8;
    Ifo=1.6;
    
    Nx = 150;
    Ny = 150;
    
    dx = Lx/Nx;
    dy = Ly/Ny;
    
    dt = 1e-8;
    
    val_cut = 1e-6;
   
    /*
    xc1d=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(1,Nx-1,xc1d);
    yc1d=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(1,Ny-1,yc1d);
    for(i=0;i<Nx-1;i++)
    {
        xc1d->data[0][i]=(0.5+(double)i)*dx;
    }
    for(i=0;i<Ny-1;i++)
    {
        yc1d->data[0][i]=(0.5+(double)i)*dy;
    }
    
    matrix *xc1d_t=0;
    xc=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nx-1,Ny-1,xc);
    yc=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nx-1,Ny-1,yc);
    xc1d_t=(matrix*)malloc(sizeof(matrix));
    matrix_create(Nx-1, 1,xc1d_t);
    matrix_transpose(xc1d,xc1d_t);
    matrix_replication(xc1d_t,xc,1,Ny-1);
    matrix_replication(yc1d,yc,Nx-1,1);
    matrix_free(xc1d_t);
    */
    
    // field vars
    phi_s=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nx,Ny,phi_s);
    phi_p=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nx,Ny,phi_p);
    
    chp_p=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nx,Ny,chp_p);
    chp_s=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nx,Ny,chp_s);
    
    dPsi_ds=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nx,Ny,dPsi_ds);
    dPsi_dp=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nx,Ny,dPsi_dp);
    
    //初始化场
    initialization();
    
    //时间积分
    himp_s=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nx,Ny,himp_s);
    himp_p=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nx,Ny,himp_p);
    
    hexp_s=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nx,Ny,hexp_s);
    hexp_p=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nx,Ny,hexp_p);
    hexp_s_np=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nx,Ny,hexp_s_np);
    hexp_p_np=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nx,Ny,hexp_p_np);
    
    double error=0;
    error=matrix_max(phi_p);
    int it=1;
    
    matrix *phi_p_np, *phi_s_np,*error_s,*error_p;
    phi_p_np=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nx,Ny,phi_p_np);
    phi_s_np=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nx,Ny,phi_s_np);
    error_s=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nx,Ny,error_s);
    error_p=(matrix*)malloc(sizeof(matrix));
    matrix_zeros(Nx,Ny,error_p);
    
    while(it<=400000)
    {
        for(i=0;i<Nx;i++)
            for(j=0;j<Ny;j++)
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
        
    for(i=0;i<Nx;i++)
    {
        for(j=0;j<Ny;j++)
        {
            fprintf(fp1,"%e ",phi_s->data[i][j]);
            fprintf(fp2,"%e ",phi_p->data[i][j]);

        }
        fprintf(fp1,"\n");
        fprintf(fp2,"\n");
    }
    matrix_free(phi_p_np);
    matrix_free(phi_s_np);
    matrix_free(error_p);
    matrix_free(error_s);
    return 0;
}
