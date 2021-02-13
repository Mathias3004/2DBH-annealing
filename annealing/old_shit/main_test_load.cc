#include <stdio.h>
#include <cmath>
#include <iostream>
#include <iomanip>
# include "itensor/all.h"
#include "itensor/mps/siteset.h"
#include "itensor/util/parallel.h"

#include "boson.h"
#include "entropy.h"

using namespace std;
using namespace itensor;

int main ( int argc, char *argv[] )
{
            //---------------------------------------------------------
    //----------------------initialize MPI---------------------
    Environment env(argc,argv);
    
    //---------------------------------------------------------
    //---------parse parameters from input file ---------------
    
    string folder = string(argv[1]);
    
    auto input = InputGroup(folder + "/input","input");
    
    //MPS parameters
    auto cutoff = input.getReal("cutoff",1E-8);
    auto Maxm = input.getInt("Maxm",200);
    
    //system parameters
    auto N = input.getInt("N"); //
    auto Nx = input.getInt("Nx",N);
    auto Ny = input.getInt("Ny",N);
    auto nfock = input.getInt("nfock",6); //
    auto filling = input.getInt("filling", 1);
    auto periodic_x = input.getInt("periodic_x", 0) == 1;
    auto periodic_y = input.getInt("periodic_y", 0) == 1;
    
    // BH model parameters
    auto U = input.getReal("U",-1.);
    auto Jmax = input.getReal("Jmax",0.15);
    auto NJ = input.getInt("NJ", 20);
    
    //annealing parameters
    auto tmin = input.getReal("tmin", 1./abs(U));
    auto tmax = input.getReal("tmax", 200./abs(U));
    auto Nt = input.getInt("Nt", env.nnodes());
    
    auto dt = input.getReal("dt",0.01/abs(U));
    
    // Make args
    auto args = Args("Cutoff=",cutoff,"Maxm=",Maxm);
    args.add("nfock=",nfock);
    
    //Setup Siteset
    int N_site = Nx*Ny;
    double t_ev = 7.;
    double J = 0.052500;
    
    
    Boson sites(N_site,args);
    string file = folder + "/sites_ts_" + to_string(t_ev) + "_J_" + to_string(J);
    cout << "test00" << endl;
    readFromFile("store/run1/sites_ts_7.000000_J_0.052500",sites);
    //readFromFile(folder + "/input",sites);
    cout << "test01" << endl;
    
    Boson sites_op(N_site, args);

   int mid_i = (int) ((N-1)/2*N + (N+1)/2);

    // density correlations 
    /*auto ampoxx_dc = AutoMPO(sites);
    ampoxx_dc += "N",mid_i, "N",mid_i;
    IQMPO dens_corr = IQMPO(ampoxx_dc);
    cout << "test1" << endl; */
    

    
    IQMPS psi(sites);
    file = folder + "/psi_ts_" + to_string(t_ev) + "_J_" + to_string(J);
    readFromFile<IQMPS>(file,psi);
    

    cout << "test2" << endl;
    psi.position(mid_i);
    cout << "test3" << endl;
    
    auto ket = toITensor(psi.A(mid_i));
    ket *= delta(dag(sites(mid_i)), sites_op(mid_i));
    cout << "test4" << endl;
    auto bra = dag(prime(ket,Site));
    cout << "test5" << endl;
    auto Nop = toITensor(sites_op.op("N",mid_i));
    //cout << "test6" << endl;
    //cout << real(overlapC(psi, dens_corr, psi)) << endl;
    PrintData(ket);
    PrintData(bra);
    PrintData(Nop);
    PrintData(bra*Nop);
     PrintData(bra*Nop*ket);
     cout << (bra*Nop*ket).real() << endl;
    cout << "test8" << endl; 
    return 0;
}
    
    
    
    
    
    