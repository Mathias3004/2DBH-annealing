#include "itensor/all.h"
#include "itensor/util/parallel.h"
#include "boson.h"

using namespace std;
using namespace itensor;

#ifndef BH2D
#define BH2D

AutoMPO create_H(Boson sites, int Nx, int Ny, double J,double U, bool periodic_x, bool periodic_y)
{
    
    AutoMPO H_eff(sites);
    int N_site = sites.N();
    
    // hopping in 2D, bulk
    for(int ix = 1; ix <= Nx; ix++)
    {
        int Rx = (ix-1)*Nx;
        int Rxp1 = ix*Nx;
        for(int iy = 1; iy <= Ny; iy++){
            // y-hopping
            if (iy<Ny){
                H_eff += -J, "b+",Rx+iy, "b-",Rx+iy+1;
                H_eff += -J, "b-",Rx+iy, "b+",Rx+iy+1;
            }
            // x-hopping
            if (ix<Nx){
                H_eff += -J, "b+",Rx+iy, "b-",Rxp1+iy;
                H_eff += -J, "b-",Rx+iy, "b+",Rxp1+iy;
            }
        }
    }
    // add periodic terms if necessary
    if (periodic_x){
        for(int iy = 1; iy <= Ny; iy++){
            H_eff += -J, "b+",iy, "b-",(Nx-1)*Ny+iy;
            H_eff += -J, "b-",iy, "b+",(Nx-1)*Ny+iy;
        }
    }
    if (periodic_y){
        for(int ix = 1; ix <= Nx; ix++){
            H_eff += -J, "b+",(ix-1)*Nx+1, "b-",(ix-1)*Nx+Ny;
            H_eff += -J, "b-",(ix-1)*Nx+1, "b+",(ix-1)*Nx+Ny;
        }
    }

    
    // local interaction terms of Hamiltonian
    for(int i = 1; i <= N_site; i++)
    {
        // interaction
        H_eff += U/2., "N(N-1)", i;
    }
    
    return H_eff;
}


#endif