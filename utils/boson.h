#ifndef __BOSON_H
#define __BOSON_H
#include <algorithm>
#include "itensor/mps/siteset.h"
#include <math.h> 
using namespace std;

namespace itensor {

class Boson : public SiteSet
    {
    public:

    Boson() { }

    Boson(int N, 
            Args const& args = Args::global());

    };


class BosonSite
    {
    IQIndex s;
    int nfock;
    public:

    BosonSite() { }

    BosonSite(IQIndex I) : s(I) { }

    BosonSite(int n, Args const& args = Args::global())
        {
        nfock = args.getInt("nfock",10);
        string name = nameint("site=",n);
        vector<IndexQN> iq(nfock+1);
        for(int n = 0; n <= nfock; n++)
            {
            iq[n] = IndexQN(Index(to_string(n),1,Site),QN("Nb=",n));
            }
        s = IQIndex(name, std::move(iq));
        }

    IQIndex
    index() const { return s; }

    IQIndexVal
    state(std::string const& state)
        {
        auto n = stoi(state);
        if(n <= nfock)
            {
            return s(n+1);
            }
        else
            {
            Error("State " + state + " not recognized");
            }
        return IQIndexVal{};
        }
 
    IQTensor
    op(std::string const& opname,
       Args const& args) const
        {
        auto sp  = prime(s);

        auto Op  = IQTensor(dag(s),sp);
        if(opname == "b+" || opname == "ad")
            {
            for(int n = 0; n <= nfock-1; n++)
                {
                int lab = n+1; 
                int labp = n+2; 
                Op.set(s(lab), sp(labp),std::sqrt(n+1));
                }
            }
        else if(opname == "b-"|| opname == "a")
            {
            for(int n = 0; n <= nfock-1; n++)
                {
                int lab = n+2; 
                int labp = n+1; 
                Op.set(s(lab), sp(labp),std::sqrt(n+1));
                }
            }
        else if(opname == "N")
            {
            for(int n = 0; n <= nfock; n++)
                {
                int lab = n+1; 
                int labp = n+1; 
                Op.set(s(lab), sp(labp), n);
                }
            }
   
        else if(opname == "N-1")
        {
            for(int n = 1; n <= nfock; n++)
            {
                int lab = n+1;
                int labp = n+1;
                Op.set(s(lab), sp(labp), n-1.);
            }
        }
        else if(opname == "N+1")
        {
            for(int n = 0; n <= nfock-1; n++)
            {
                int lab = n+1;
                int labp = n+1;
                Op.set(s(lab), sp(labp), n+1.);
            }
        }
        else if(opname == "N(N-1)")
        {
            for(int n = 1; n <= nfock; n++)
            {
                int lab = n+1;
                int labp = n+1;
                Op.set(s(lab), sp(labp), n*(n-1.));
            }
        }
            
        else if(opname == "N^2")
        {
            for(int n = 0; n <= nfock; n++)
            {
                int lab = n+1;
                int labp = n+1;
                Op.set(s(lab), sp(labp), n*n);
            }
        }
        
        else if(opname == "Nred2")
        {
            for(int n = 0; n <= min(2,nfock); n++)
            {
                int lab = n+1;
                int labp = n+1;
                Op.set(s(lab), sp(labp), n);
            }
        }

       else if(opname == "b+^2" || opname == "ad^2")
        {
            for(int n = 0; n <= nfock-2; n++)
            {
                int lab = n+1;
                int labp = n+3;
                Op.set(s(lab), sp(labp),std::sqrt((n+2)*(n+1)));
            }
        }
        else if(opname == "b-^2"|| opname == "a^2")
        {
            for(int n = 0; n <= nfock-2; n++)
            {
                int lab = n+3;
                int labp = n+1;
                Op.set(s(lab), sp(labp),std::sqrt((n+2)*(n+1)));
            }
        }
        
        else if(opname == "Id" || opname == "ID")
            {
            for(int n = 0; n <= nfock; n++)
                {
                int lab = n+1; 
                int labp = n+1; 
                Op.set(s(lab), sp(labp), 1);
                }
            }
        else
            {
            Error("Operator \"" + opname + "\" name not recognized");
            }

        return Op;
        }
    };

inline Boson::
Boson(int N, 
        Args const& args)
    {
    auto sites = SiteStore(N);

    for(int j = 1; j < N+1; ++j)
        {
        sites.set(j,BosonSite(j,args));
        }

    SiteSet::init(std::move(sites));
    }

 }//namespace itensor

#endif
