#include <stdio.h>
#include <cmath>
#include <iostream>
#include <iomanip>
# include "itensor/all.h"
#include "itensor/mps/siteset.h"
#include "itensor/util/parallel.h"

#include "boson.h"
#include "entropy.h"
#include "MPSfunctions.h"

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
    auto Nt = input.getInt("Nt", env.nnodes()-1);
    
    auto dt = input.getReal("dt",0.01/abs(U));
    
    // Make args
    auto args = Args("Cutoff=",cutoff,"Maxm=",Maxm);
    args.add("nfock=",nfock);
    
    //Setup Siteset
    int N_site = Nx*Ny;
    auto sites = Boson(N_site, args);

    
    //****************************************************************
    // OBSERVABLES
    //****************************************************************
    
    int mid_i = (int) ((N-1)/2*N + (N+1)/2);

    // density correlations 
    auto ampoxx_dc = AutoMPO(sites);
    ampoxx_dc += "N",mid_i, "N",mid_i;
    IQMPO dens_corr = IQMPO(ampoxx_dc);

    // densities 
    auto ampoxx_n = AutoMPO(sites);
    ampoxx_n += "N",mid_i;
    IQMPO density = IQMPO(ampoxx_n);
    
    // correlation
    auto ampoxx_c = AutoMPO(sites);
    ampoxx_c += "ad",mid_i,"a",mid_i+1;
    IQMPO quadratic_corr = IQMPO(ampoxx_c);
    
    // density correlations with level 0,1,2 only
    auto ampoxx_dc2 = AutoMPO(sites);
    ampoxx_dc2 += "Nred2",mid_i, "Nred2",mid_i;
    IQMPO dens2_corr = IQMPO(ampoxx_dc2);

    // densities 0,1,2 only
    auto ampoxx_n2 = AutoMPO(sites);
    ampoxx_n2 += "Nred2",mid_i;
    IQMPO density2 = IQMPO(ampoxx_n2);

    
    // J values for rank
    double time_diff = (tmax-tmin)/(Nt-1);
    double t_ev = tmin + env.rank()*time_diff;
    
    
    // start loop
    double deltaJ_save = Jmax/(NJ);
    int count = 0;
    vector<double> S;
    vector<double> n;
    vector<double> c;
    vector<double> dc;
    vector<double> n2;
    vector<double> dc2;
    
    for (int iJ = 0; iJ<= NJ; iJ++){
        
        double J = deltaJ_save*iJ;
        
        string file_MPS;
        string file_sites;
        if (env.rank() == Nt){
            double sJ;
            if (iJ == 0){ sJ = J*( -Jmax/abs(Jmax) );}
            else{ sJ = J;}
            file_MPS = folder + "/psi_DMRG_J_" + to_string(sJ);
            file_sites = folder + "/sites_DMRG_J_" + to_string(sJ);
        }
        else{
            double sJ;
            if (iJ == 0){ sJ = J*( Jmax/abs(Jmax) );}
            else{ sJ = J;}
            file_MPS = folder + "/psi_ts_" + to_string(t_ev) + "_J_" + to_string(sJ);
            file_sites = folder + "/sites_ts_" + to_string(t_ev);
        }
        
        
         std::ifstream infile(file_MPS);
        
        if (infile.good()){
        
            IQMPS psi;
            psi = readMPS(sites, file_sites, file_MPS);
            cout << "rank " + to_string(env.rank()) + ": successfully loaded file " << file_MPS <<  endl;
            
            // normalize
            psi.position(mid_i);
            
            // observables
            S.push_back(S_vn_single_site(psi,mid_i));
            n.push_back(real(overlapC(psi, density, psi)));
            dc.push_back(real(overlapC(psi, dens_corr, psi)));
            c.push_back(real(overlapC(psi, quadratic_corr, psi)));
            n2.push_back(real(overlapC(psi, density2, psi)));
            dc2.push_back(real(overlapC(psi, dens2_corr, psi)));
        }
        else{
            cout << "rank " + to_string(env.rank()) + ": NOT successfully loaded file " << file_MPS <<  endl;
            // observables
            S.push_back(-1.);
            n.push_back(-1.);
            c.push_back(-1.);
            dc.push_back(-1.);
            n2.push_back(-1.);
            dc2.push_back(-1.);
        }

        
        // let all nodes finish
        env.barrier();
        if (env.rank()==0){
            cout << endl << "**************************************************************" <<endl;
            cout << "\tJ value " + to_string(iJ+1) + "/" + to_string(NJ+1) + " finished" << endl;
            cout << "**************************************************************" <<endl << endl;
        }
        
    }
    
    if (env.rank()==0) cout << "Saving data... ";
    // save data to file
    ofstream n_file;
    ofstream c_file;
    ofstream dc_file;
    ofstream n2_file;
    ofstream dc2_file;
    ofstream S_file;
    
    for (int i = 0; i<env.nnodes(); i++){
        if (i==env.rank()){
            if (i == 0){
                n_file.open(folder + "/n_mid.txt");
                c_file.open(folder + "/c_mid.txt");
                dc_file.open(folder + "/dc_mid.txt");
                n2_file.open(folder + "/n2_mid.txt");
                dc2_file.open(folder + "/dc2_mid.txt");
                S_file.open(folder + "/S_mid.txt");  
            }
            else {
                n_file.open(folder + "/n_mid.txt", ofstream::out | ofstream::app);
                c_file.open(folder + "/c_mid.txt", ofstream::out | ofstream::app);
                dc_file.open(folder + "/dc_mid.txt", ofstream::out | ofstream::app);
                n2_file.open(folder + "/n2_mid.txt", ofstream::out | ofstream::app);
                dc2_file.open(folder + "/dc2_mid.txt", ofstream::out | ofstream::app);
                S_file.open(folder + "/S_mid.txt", ofstream::out | ofstream::app);  
            }
            for (int iJ=0; iJ<NJ+1; iJ++){
                n_file << n[iJ] << " ";
                c_file << c[iJ] << " ";
                dc_file << dc[iJ] << " ";
                n2_file << n2[iJ] << " ";
                dc2_file << dc2[iJ] << " ";
                S_file << S[iJ] << " ";
            }
            n_file << endl;
            c_file << endl;
            dc_file << endl;
            n2_file << endl;
            dc2_file << endl;
            S_file << endl;
            
        }
        // halt
        env.barrier();
    }
    
    if (env.rank()==0) cout << "DONE" << endl << endl;
    return 0;
}