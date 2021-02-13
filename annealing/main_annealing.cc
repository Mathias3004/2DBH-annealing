//
//  main.cpp
//  annealing 2D Bose Hubbard model using MPI
//
//  Created by Mathias on 7/8/2020
//

#include <stdio.h>
#include <cmath>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdint>
//#include <filesystem>
#include "itensor/all.h"
#include "itensor/util/parallel.h"

#include "boson.h"
#include "2DBH_ham.h"
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
    auto input = InputGroup(argv[1],"input");
    
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
    
    auto MPO_apply = input.getString("MPO_apply", "exact");

    //Outpu folder MPS
    auto output_folder = input.getString("output_folder","data"); //
    
    //*********************************************************
    // START 
    //*********************************************************
    
    // check if folder exists, otherwise make it
    //if (!filesystem::exists(output_folder)) filesystem::create_directory(output_folder);
    
    // copy input file to folder for later
    if (env.rank() == 0){ 
        system(("cp " + (string)argv[1] + " " + output_folder + "/input").c_str());
        ofstream info;
        info.open(output_folder + "/info.txt");
        info << "np " << env.nnodes();
    }
    
    
    // Make args
    auto args = Args("Cutoff=",cutoff,"Maxm=",Maxm);
    args.add("nfock=",nfock);
    
    //Setup Siteset
    int N_site = Nx*Ny;
    auto sites = Boson(N_site, args);
    // save sites
    //writeToFile( output_folder + "/sites", sites );
 
    
    
    
    //-----------------------------------------------------
    //---------Initial Condition---------------------------
    // initial state with uniform filling
    
    auto state = InitState(sites);
    for(int i = 1; i <= N_site; ++i)
    {
        state.set(i,to_string(filling));
    }
    auto psi = IQMPS(state);
    
    //**********************************************************
    // TIME EVOLUTION
    //**********************************************************

   // for the time evolution between different J
    double time_diff = (tmax-tmin)/(Nt-1);
    double t_ev = tmin + env.rank()*time_diff;
    double t_interval_J = t_ev/NJ;
    
    int N_t_step = ceil(t_interval_J/dt);
    double dt_eff = t_interval_J/(double)N_t_step;
    double dJ = Jmax/N_t_step/NJ;
    
    //cout << "Rank " << env.rank() << " N_t_step " << N_t_step << ", dt_eff " << dt_eff << endl;

    
    std::clock_t start;
    double J = 0.;
    
    // save initial state
    writeToFile(output_folder + "/psi_ts_" + to_string(t_ev)  + "_J_" + to_string(J),psi);
    
    // save sites of 
    writeToFile(output_folder + "/sites_ts_" + to_string(t_ev), sites);
    
    for (int iJ = 1; iJ<=NJ ; iJ++){
        double time_ampo = 0.;
        double time_psi = 0.;
        for (int it = 1; it<= N_t_step; it++){
        
            // update J
            J += dJ;
    
            // set up H and exponentiate
            start = clock();
            auto ampo = create_H(sites, Nx, Ny, J, U, periodic_x, periodic_y);

            auto expH = toExpH<IQTensor>(ampo,dt_eff*Cplx_i);
            time_ampo += (clock() - start)/ (double) CLOCKS_PER_SEC;

        
            start = clock();
            if (MPO_apply == "exact"){
                psi = exactApplyMPO(expH,psi,args);}
            else{
                psi = fitApplyMPO(expH,psi,args);
            }
            psi.normalize();
            time_psi += (clock() - start)/ (double) CLOCKS_PER_SEC;
        }
        
        // save MPS
      
            
        writeToFile(output_folder + "/psi_ts_" + to_string(t_ev)  + "_J_" + to_string(J),psi);
        //writeToFile(output_folder + "/sites_ts_" + to_string(t_ev)  + "_J_" + to_string(J),sites);
            
        cout << "Rank " << env.rank() << " saved state " << iJ << "/" << NJ;
        cout << ", time_ampo: " << time_ampo << ", time_psi: " << time_psi << ", max BD: " << maxM(psi)  << endl;
            
        time_ampo = 0.;
        time_psi = 0.;
            
    }
        
    
    cout << "Rank " << env.rank() << " finished" << endl;
    
    
    // write the states and sites to folder
    
    /*for (int iJ = 0; iJ<NJ; iJ++){
        writeToFile(output_folder + "/psi_ts_" + to_string(t_ev)  + "_J_" to_string(iJ*deltaJ_save),vpsi[iJ]);
    }*/
    
    
   /*env.barrier();
    cout << "Rank " << U_sel << ": collecting observables...";
    double dE = en1-en0;
    
    vector<double> density_val;
    vector<double> dens_corr_val;
    vector<double> quad_corr_val;
    vector<double> S_vn_val;
    for (int i=0; i<N; i++){
        density_val.push_back(real(overlapC(psi, density[i], psi)));
        dens_corr_val.push_back(real(overlapC(psi, dens_corr[i], psi)));
        quad_corr_val.push_back(real(overlapC(psi, quad_corr[i], psi)));
        S_vn_val.push_back(S_vn(psi, i*N));
        env.barrier();
        if (U_sel==0) cout << "point " << i << " finished" << endl;
    }

    
    cout << "\n\n-----------------------------------" << endl;
    cout << "Rank " << U_sel <<  " finished" << endl;  
    cout << "-----------------------------------" << endl;
    
    // wait for all processes to finish
    env.barrier();
    
    
    if (U_sel==0) cout << "Writing..." << endl;
     
    // copy data   
    for (int iU = 0; iU<NU; iU++){
        
        // select right rank
        if (iU==U_sel){
            
            cout << "Writing rank " << U_sel << "..." << endl;
            
            ofstream U_file;
            ofstream quad_corrs_file;
            ofstream dens_corrs_file;
            ofstream densities_file;
            ofstream entropies_file;
            if (iU == 0){
                
                U_file.open(output_folder + "/Uval.txt");
                quad_corrs_file.open(output_folder + "/quad_corrs.txt");
                dens_corrs_file.open(output_folder + "/dens_corrs.txt");
                densities_file.open(output_folder + "/densities.txt");
                entropies_file.open(output_folder + "/entropies.txt");  
            }
            else {
                U_file.open(output_folder + "/Uval.txt", ofstream::out | ofstream::app);
                quad_corrs_file.open(output_folder + "/quad_corrs.txt", ofstream::out | ofstream::app);
                dens_corrs_file.open(output_folder + "/dens_corrs.txt", ofstream::out | ofstream::app);
                densities_file.open(output_folder + "/densities.txt", ofstream::out | ofstream::app);
                entropies_file.open(output_folder + "/entropies.txt", ofstream::out | ofstream::app);  
            }
        
            //writeToFile(output_folder + "/psi_" + to_string(vU[U_sel]),psi);
            U_file << vU[U_sel] << " " << dE << endl;
            U_file.close();
            for (int i=0; i<N; i++){
                densities_file << density_val[i] << " ";
                dens_corrs_file << dens_corr_val[i] << " ";
                quad_corrs_file << quad_corr_val[i] << " ";
                entropies_file << S_vn_val[i] << " ";
            }
            quad_corrs_file << endl;
            dens_corrs_file << endl;
            densities_file << endl;
            entropies_file << endl;
            
            quad_corrs_file.close();
            dens_corrs_file.close();
            densities_file.close();
            entropies_file.close();
            
            cout << "Finished" << endl << endl;
        }
        
        // wait for all processes to finish
        env.barrier();
    } */
    
    return 0;
}
    
    
    
    
    
    
    
    
    
    


