//
//  main.cpp
//  2D DMRG for Bose Hubbard model
//
//  Created by Mathias on 7/8/2020
//

#include <stdio.h>
#include <cmath>
#include <iostream>
#include <iomanip>
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
    auto Maxm = input.getInt("Maxm",50);
    
    //system parameters
    auto N = input.getInt("N"); //
    auto Nx = input.getInt("Nx",N);
    auto Ny = input.getInt("Ny",N);
    auto nfock = input.getInt("nfock",6); //
    auto filling = input.getInt("filling", 1);
    auto periodic_x = input.getInt("periodic_x", 0) == 1;
    auto periodic_y = input.getInt("periodic_y", 0) == 1;
    
    // BH model parameters
    auto U = input.getReal("U",1.);
    auto Jmin = input.getReal("Jmin",0.);
    auto Jmax = input.getReal("Jmax", 0.15);
    auto NJ = env.nnodes();
    
    //DMRG parameters
    auto N_sweeps = input.getInt("N_sweeps",5);

    //Outpu folder MPS
    auto output_folder = input.getString("output_folder","data"); //
    
    
    // Make args
    auto args = Args("Cutoff=",cutoff,"Maxm=",Maxm);
    args.add("nfock=",nfock);
    
    //Setup Siteset
    int N_site = Nx*Ny;
    auto sites = Boson(N_site, args);
    
    // write sites to folder
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
    auto psi1 = IQMPS(state);

   
    
    
    //-----------------------------------------------------
    // DMRG sweeps ----------------------------------------
    auto sweeps = Sweeps(5);
    sweeps.maxm() = 10,40,200,300,300;
    sweeps.cutoff() = cutoff;
    
    vector<double> vJ(NJ);
    double dJ = (Jmax-Jmin)/(NJ-1);
    for (int iJ = 0; iJ<NJ; iJ++){
        vJ[iJ] = Jmin +iJ*dJ;
    }
    
    //-----------------------------------------------------
    // Transition matrix element
    auto ampo = create_H(sites, Nx, Ny, Jmax, 0., periodic_x, periodic_y);
    auto dHds = IQMPO(ampo);
    
    
    // select the J
    int J_sel = env.rank();
    
    cout << "Rank " << env.rank() << " running with tunneling J=" << vJ[J_sel] << endl;
    
    // set up H
    ampo = create_H(sites, Nx, Ny, vJ[J_sel], U, periodic_x, periodic_y);
    auto H = IQMPO(ampo);
    
    auto en0 = dmrg(psi,H,sweeps,{"Quiet",true});
    
    auto wfs = std::vector<IQMPS>(1);
    wfs.at(0) = psi;
    
    auto en1 = dmrg(psi1,H,wfs,sweeps,{"Quiet=",true,"Weight=",20.0});
    
    // transition amplitude |0> to |1>
    auto TME = overlap(psi1,dHds,psi);
    
    cout << "Rank " << J_sel << ": DMRG done" << endl;
    
    env.barrier();
    cout << "Rank " << J_sel << ": collecting observables..." << endl;
    //double dE = en1-en0;
    
    
    cout << "\n\n-----------------------------------" << endl;
    cout << "Rank " << J_sel <<  " finished" << endl;  
    cout << "-----------------------------------" << endl;
    
    // wait for all processes to finish
    env.barrier();
    
    
    if (J_sel==0) cout << "Writing..." << endl;
    
    // save MPS
    writeToFile(output_folder + "/psi_DMRG" + "_J_" + to_string(-vJ[J_sel]),psi);
    // save sites of 
    writeToFile(output_folder + "/sites_DMRG" + "_J_" + to_string(-vJ[J_sel]), sites);
    
    // copy energies  
    ofstream E_file;
    for (int i = 0; i<NJ; i++){
        if (i==env.rank()){
            if (i == 0){
                E_file.open(output_folder + "/E.txt");
            }
            else{
                E_file.open(output_folder + "/E.txt", ofstream::out | ofstream::app);
            }
            
            E_file << en0 << " " << en1 << " " << TME << endl;
            
        }
        
        env.barrier();
    }
     
  
    
    return 0;
}
    
    
    
    
    
    
    
    
    
    


