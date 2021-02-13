#include "itensor/all.h"
#include "itensor/util/parallel.h"
#include "boson.h"

using namespace std;
using namespace itensor;

#ifndef MPSf
#define MPSf

IQMPS readMPS(SiteSet sites, string site_name, string psi_name)
{
    SiteSet sites_read1;
    int N = sites.N();
    readFromFile(site_name,sites_read1);
    IQMPS psi(sites_read1);
    readFromFile(psi_name,psi);
    for(int i = 1; i <= N; i++)
    {
        psi.Aref(i) *= delta(dag(sites_read1(i)), sites(i));
    }
    
    psi.position(1);
    return psi;
}

//double expect_val_single_site(IQMPS psi, IQTensor)



#endif