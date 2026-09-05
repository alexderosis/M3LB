#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include "VTKWriter.h"
using namespace std;

const bool plot_vtk = true;

typedef Kokkos::View<double*> View1DArray;
typedef Kokkos::View<double**> View2DArray;
typedef Kokkos::View<double***> View3DArray;
typedef Kokkos::View<double****> View4DArray;

struct D3Q27{
  // lattice velocities
    static constexpr double cs2 = 1./3.;
    static constexpr int dim = 3;
    static constexpr int np = 27;
    const int cx[np];
    const int cy[np];
    const int cz[np];
    const int opp[np];
    const double wf[np];

    D3Q27():
        // 0  1   2  3   4  5   6  7   8   9  10 11  12  13  14 15  16  17  18 19  20  21  22  23  24  25  26
        cx{0, 1, -1, 0,  0, 0,  0, 1, -1,  1, -1, 1, -1,  1, -1, 0,  0,  0,  0, 1, -1,  1, -1,  1, -1,  1, -1},
        cy{0, 0,  0, 1, -1, 0,  0, 1,  1, -1, -1, 0,  0,  0,  0, 1, -1,  1, -1, 1,  1, -1, -1,  1,  1, -1, -1},
        cz{0, 0,  0, 0,  0, 1, -1, 0,  0,  0,  0, 1,  1, -1, -1, 1,  1, -1, -1, 1,  1,  1,  1, -1, -1, -1, -1},
       opp{0, 2,  1, 4,  3, 6,  5,10,  9,  8,  7,14, 13, 12, 11,18, 17, 16, 15, 26, 25, 24,23, 22, 21, 20, 19},
        // Weight factors for lattice directions
        wf{8./27., 2./27., 2./27., 2./27., 2./27., 2./27., 2./27.,
            1./54., 1./54., 1./54., 1./54., 1./54., 1./54., 1./54., 1./54., 1./54., 1./54., 1./54., 1./54.,
             1./216., 1./216., 1./216., 1./216., 1./216., 1./216., 1./216., 1./216.} {}
};

struct Params{
    static constexpr double rho0 = 1;
    static constexpr double C = 10.;
    static constexpr double M = 10.;
    static constexpr double alpha = 1e-3;
    static constexpr double T = 200.;
    static constexpr double phi_b = 1.;
    static constexpr double phi_t = 0.;
    static constexpr double DeltaPhi = phi_b-phi_t;
    static constexpr double U_ref = 0.01;
    static constexpr int nx = 200;
    static constexpr int ny = nx/2;
    static constexpr int nz = 1;
    static constexpr double Kappa = U_ref*ny/DeltaPhi;
    static constexpr double permittivity = M*M*Kappa*Kappa*rho0;
    static constexpr double ni = permittivity*DeltaPhi/Kappa/T;
    static constexpr double tau = ni*3.+0.5;
    static constexpr double omega = 1./tau;
    static constexpr double omega1 = 1.-omega;
    static constexpr double T_ref = ny/U_ref;
    static constexpr double q0 = C*permittivity*DeltaPhi/ny/ny;
    static constexpr double charge_diffusion = alpha*Kappa*DeltaPhi;
    static constexpr double tau_q = charge_diffusion*3.+0.5;
    static constexpr double omega_q = 1./tau_q;
    static constexpr double omega_q1 = 1.-omega_q;

    static constexpr double beta_phi = 0.3;
    static constexpr double tau_phi = 3.*beta_phi+0.5;
    static constexpr double omega_phi = 1./tau_phi;
    static constexpr double omega_phi1 = 1.-omega_phi;

    static constexpr int nsteps = (int)(100*T_ref);
    static constexpr int n_out = (int)(1*T_ref);
};

void writeData(const std::string& data_path, double i, double T_ref,
                double velocity, bool first_write) {

    // If it's the first write, open in trunc mode to clear previous contents
    std::ios_base::openmode mode = first_write ? std::ios::trunc : std::ios::app;
    
    std::ofstream data_output(data_path, mode);
    if (!data_output.is_open()) {
        std::cerr << "Failed to open file: " << data_path << std::endl;
        return;
    }

    // Write formatted data
    data_output << i/T_ref << "\t"
                << velocity << "\n";

    data_output.close();
}

void initial_state(View3DArray rho, View3DArray u, View3DArray v, View3DArray w, View4DArray f1, View4DArray f2,
                    View4DArray h1, View4DArray h2, View4DArray g1, View4DArray g2, View3DArray q, View3DArray phi, 
                    View3DArray Ex, View3DArray Ey, View3DArray Ez, const D3Q27 lattice, const Params parameters) {

    // Kokkos lambda for initialization
    Kokkos::parallel_for("InitializeState",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {parameters.nx, parameters.ny, parameters.nz}),
      KOKKOS_LAMBDA(const int x, const int y, const int z) {

        double R = parameters.rho0;
        double U = 0.;
        double V = 0.;
        double W = 0.;
        double Q = 0.;
        double Phi = 0.;
        double EX, EY, EZ;
        if(y==0)
        {
          Q = parameters.q0;
          Phi = parameters.phi_b;
        }
        if(y==parameters.ny-1)
        {
          Q = 0.;
          Phi = parameters.phi_t;
        }

        rho(x, y, z) = R;
        u(x, y, z) = U;
        v(x, y, z) = V;
        w(x, y, z) = W;
        q(x, y, z) = Q;
        phi(x, y, z) = Phi;
        Ex(x, y, z) = EX = 0.;
        Ey(x, y, z) = EY = 0.;
        Ez(x, y, z) = EZ = 0.;

        double first_order, second_order;

        for (int k = 0; k < lattice.np; k++) {
            first_order = U*lattice.cx[k] + V*lattice.cy[k] + W*lattice.cz[k];
            f1(x, y, z, k) = f2(x, y, z, k) = lattice.wf[k] * R  * (1. + 3.*first_order + 4.5*pow(first_order,2) -1.5*(U*U+V*V+W*W));

            first_order = (parameters.Kappa*EX+U)*lattice.cx[k] + (parameters.Kappa*EY+V)*lattice.cy[k] + (parameters.Kappa*EZ+W)*lattice.cz[k];
            h1(x, y, z, k) = h2(x, y, z, k) = lattice.wf[k] * Q  * (1. + 3.*first_order + 4.5*pow(first_order,2)
                            - 1.5*(pow(parameters.Kappa*EX+U,2)+pow(parameters.Kappa*EY+V,2)+pow(parameters.Kappa*EZ+W,2)));

            g1(x, y, z, k) = g2(x, y, z, k) = lattice.wf[k] * Phi;
        }
    }
    );
    Kokkos::fence(); 
}

void copy_phi(View3DArray phi, View3DArray phi_old, const Params parameters) {

    // Kokkos lambda for initialization
    Kokkos::parallel_for("copy_phi",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {parameters.nx, parameters.ny, parameters.nz}),
    KOKKOS_LAMBDA(const int x, const int y, const int z) {

        phi_old(x, y, z) = phi(x, y, z);
    }
    );
    Kokkos::fence(); 
}

void algoLB(View4DArray f1, View4DArray f2, View4DArray h1, View4DArray h2, View4DArray g1, View4DArray g2, View3DArray rho, View3DArray u, View3DArray v, View3DArray w,
    View3DArray Ex, View3DArray Ey, View3DArray Ez, View3DArray q, View3DArray phi, View3DArray phi_old,
    const D3Q27 lattice, const Params parameters){

    // Kokkos parallel loop for the lattice Boltzmann algorithm
    Kokkos::parallel_for("LBAlgorithm",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {parameters.nx, parameters.ny, parameters.nz}),
    KOKKOS_LAMBDA(const int x, const int y, const int z) {

          Ex(x, y, z) = Ey(x, y, z) = Ez(x, y, z) = 0.;
          if(y==0)
          {
            Ex(x, y, z) = Ez(x, y, z) = 0.;
            Ey(x, y, z) = -(-1.5*phi_old(x, y, z)+2.*phi_old(x, y+1, z)-0.5*phi_old(x, y+2, z));
          }
          else if(y==parameters.ny-1)
          {
            Ex(x, y, z) = Ez(x, y, z) = 0.;
            Ey(x, y, z) = -(1.5*phi_old(x, y, z)-2.*phi_old(x, y-1, z)+0.5*phi_old(x, y-2, z));
          }
          else
          {
            if(x==0 || x==parameters.nx-1)
              Ex(x, y, z) = 0.;
            else
              Ex(x, y, z) = -0.5*(phi_old(x+1, y, z)-phi_old(x-1, y, z));
            Ey(x, y, z) = -0.5*(phi_old(x, y+1, z)-phi_old(x, y-1, z));
            Ez(x, y, z) = -0.5*(phi_old(x, y, (z+1+parameters.nz)%parameters.nz)-phi_old(x, y, (z-1+parameters.nz)%parameters.nz));
          }

        double R, U, V, W, EX, EY, EZ, Q, Phi = 0., Fx = 0., Fy = 0., Fz = 0.;
        R = U = V = W = Q = 0.0;
        for (int k=0; k<lattice.np; k++){
            R += f1(x, y, z, k);
            U += f1(x, y, z, k) * lattice.cx[k];
            V += f1(x, y, z, k) * lattice.cy[k];
            W += f1(x, y, z, k) * lattice.cz[k];
            Q += h1(x, y, z, k);
            Phi += g1(x, y, z, k);
            }
        phi(x,y,z) = Phi;
        EX = Ex(x,y,z);
        EY = Ey(x,y,z);
        EZ = Ez(x,y,z);  
        Fx = Q*EX;
        Fy = Q*EY;
        Fz = Q*EZ;
        U += 0.5*Fx;
        V += 0.5*Fy;
        W += 0.5*Fz;
        U /= R;
        V /= R;
        W /= R;

        rho(x, y, z) = R;
        u(x, y, z) = U;
        v(x, y, z) = V;
        w(x, y, z) = W;
        
        double U2 = U*U;
        double V2 = V*V;
        double W2 = W*W;

        double r4 = f1(x,y,z,7) - f1(x,y,z,8) - f1(x,y,z,9) + f1(x,y,z,10) + f1(x,y,z,19) - f1(x,y,z,20) - f1(x,y,z,21) + f1(x,y,z,22) + f1(x,y,z,23) - f1(x,y,z,24) - f1(x,y,z,25) + f1(x,y,z,26);
        double r5 = f1(x,y,z,11) - f1(x,y,z,12) - f1(x,y,z,13) + f1(x,y,z,14) + f1(x,y,z,19) - f1(x,y,z,20) + f1(x,y,z,21) - f1(x,y,z,22) - f1(x,y,z,23) + f1(x,y,z,24) - f1(x,y,z,25) + f1(x,y,z,26);
        double r6 = f1(x,y,z,15) - f1(x,y,z,16) - f1(x,y,z,17) + f1(x,y,z,18) + f1(x,y,z,19) + f1(x,y,z,20) - f1(x,y,z,21) - f1(x,y,z,22) - f1(x,y,z,23) - f1(x,y,z,24) + f1(x,y,z,25) + f1(x,y,z,26);
        double r7 = f1(x,y,z,1) + f1(x,y,z,2) - f1(x,y,z,3) - f1(x,y,z,4) + f1(x,y,z,11) + f1(x,y,z,12) + f1(x,y,z,13) + f1(x,y,z,14) - f1(x,y,z,15) - f1(x,y,z,16) - f1(x,y,z,17) - f1(x,y,z,18);
        double r8 = f1(x,y,z,1) + f1(x,y,z,2) - f1(x,y,z,5) - f1(x,y,z,6) + f1(x,y,z,7) + f1(x,y,z,8) + f1(x,y,z,9) + f1(x,y,z,10) - f1(x,y,z,15) - f1(x,y,z,16) - f1(x,y,z,17) - f1(x,y,z,18);

        double k4 = r4 - R*U*V;
        double k5 = r5 - R*U*W;
        double k6 = r6 - R*V*W;
        double k7 = r7 - R*(U2-V2);
        double k8 = r8 - R*(U2-W2);
        
        ///collide moments
        double k1 = 0.5*Fx/R;
        double k2 = 0.5*Fy/R;
        double k3 = 0.5*Fz/R;
        k4 = parameters.omega1*k4;
        k5 = parameters.omega1*k5;
        k6 = parameters.omega1*k6;
        k7 = parameters.omega1*k7;
        k8 = parameters.omega1*k8;
        double k9 = R;
        double k10 = lattice.cs2*Fx/R;
        double k11 = lattice.cs2*Fy/R;
        double k12 = lattice.cs2*Fz/R;
        double k17 = R*lattice.cs2;
        double k18 = R*lattice.cs2*lattice.cs2;
        double k23 = Fx*lattice.cs2*lattice.cs2*0.5;
        double k24 = Fy*lattice.cs2*lattice.cs2*0.5;
        double k25 = Fz*lattice.cs2*lattice.cs2*0.5;
        double k26 = R*lattice.cs2*lattice.cs2*lattice.cs2;

        double r0 = R;
        double r1 = U*R + k1;
        double r2 = V*R + k2;
        double r3 = W*R + k3;
        r4 = U*V*R + V*k1 + U*k2 + k4;
        r5 = U*W*R + W*k1 + U*k3 + k5;
        r6 = V*W*R + W*k2 + V*k3 + k6;
        r7 = (U2 - V2)*R + (2.*U)*k1 + (-2.*V)*k2 + k7;
        r8 = (U2 - W2)*R + 2.*U*k1 + (-2.*W)*k3 + k8;
        double r9 = (U2 + V2 + W2)*R + (2.*U)*k1 + 2.*V*k2 + 2.*W*k3 + k9;
        double r10 = (U*V2 + U*W2)*R + (V2 + W2)*k1 + 2.*U*V*k2 + 2.*U*W*k3 + 2.*V*k4 + 2.*W*k5 + (-U*lattice.cs2)*k7 + (-U*lattice.cs2)*k8 + ((2.*U)*lattice.cs2)*k9 + k10;
        double r11 = (V*U2 + V*W2)*R + 2.*U*V*k1 + (U2 + W2)*k2 + 2.*V*W*k3 + 2.*U*k4 + 2.*W*k6 + ((2.*V)*lattice.cs2)*k7 + (-V*lattice.cs2)*k8 + ((2.*V)*lattice.cs2)*k9 + k11;
        double r12 = (W*U2 + W*V2)*R + 2.*U*W*k1 + 2.*V*W*k2 + (U2 + V2)*k3 + 2.*U*k5 + 2.*V*k6 + (-W*lattice.cs2)*k7 + ((2.*W)*lattice.cs2)*k8 + ((2.*W)*lattice.cs2)*k9 + k12;
        double r13 = (U*V2 - U*W2)*R + (V2 - W2)*k1 + 2.*U*V*k2 + (-2.*U*W)*k3 + 2.*V*k4 + (-2.*W)*k5 + (-U)*k7 + U*k8;
        double r14 = (V*U2 - V*W2)*R + 2.*U*V*k1 + (U2 - W2)*k2 + (-2.*V*W)*k3 + 2.*U*k4 + (-2.*W)*k6 + V*k8;
        double r15 = (W*U2 - W*V2)*R + 2.*U*W*k1 + (-2.*V*W)*k2 + (U2 - V2)*k3 + 2.*U*k5 + (-2.*V)*k6 + W*k7;
        double r16 =  U*V*W*R + V*W*k1 + U*W*k2 + U*V*k3 + W*k4 + V*k5 + U*k6;
        double r17 = (U2*V2 + U2*W2 + V2*W2)*R + (2.*U*V2 + 2.*U*W2)*k1 + (2.*V*U2 + 2.*V*W2)*k2 + (2.*W*U2 + 2.*W*V2)*k3 + 4.*U*V*k4 + 4.*U*W*k5 + 4.*V*W*k6 + (- U2*lattice.cs2 + (2.*V2)*lattice.cs2 - W2*lattice.cs2)*k7 + (- U2*lattice.cs2 - V2*lattice.cs2 + (2.*W2)*lattice.cs2)*k8 + ((2.*U2)*lattice.cs2 + (2.*V2)*lattice.cs2 + (2.*W2)*lattice.cs2)*k9 + 2.*U*k10 + 2.*V*k11 + 2.*W*k12 + k17;
        double r18 = (U2*V2 + U2*W2 - V2*W2)*R + (2.*U*V2 + 2.*U*W2)*k1 + (2.*V*U2 - 2.*V*W2)*k2 + (2.*W*U2 - 2.*W*V2)*k3 + 4.*U*V*k4 + 4.*U*W*k5 + (-4.*V*W)*k6 + (W2 - U2*lattice.cs2)*k7 + (V2 - U2*lattice.cs2)*k8 + ((2.*U2)*lattice.cs2)*k9 + 2.*U*k10 + k18;
        double r19 = (U2*V2 - U2*W2)*R + (2.*U*V2 - 2.*U*W2)*k1 + 2.*U2*V*k2 + (-2.*U2*W)*k3 + 4.*U*V*k4 + (-4.*U*W)*k5 + (- U2 + V2*lattice.cs2 - W2*lattice.cs2)*k7 + (U2 + V2*lattice.cs2 - W2*lattice.cs2)*k8 + (V2*lattice.cs2 - W2*lattice.cs2)*k9 + V*k11 + (-W)*k12;
        double r20 = U2*V*W*R + 2.*U*V*W*k1 + U2*W*k2 + U2*V*k3 + 2.*U*W*k4 + 2.*U*V*k5 + U2*k6 + ((V*W)*lattice.cs2)*k7 + ((V*W)*lattice.cs2)*k8 + ((V*W)*lattice.cs2)*k9 + (W*0.5)*k11 + (V*0.5)*k12;
        double r21 = U*V2*W*R + V2*W*k1 + 2.*U*V*W*k2 + U*V2*k3 + 2.*V*W*k4 + V2*k5 + 2.*U*V*k6 + (-(2.*U*W)*lattice.cs2)*k7 + ((U*W)*lattice.cs2)*k8 + ((U*W)*lattice.cs2)*k9 + (W*0.5)*k10 + (U*0.5)*k12;
        double r22 = U*V*W2*R + V*W2*k1 + U*W2*k2 + 2.*U*V*W*k3 + W2*k4 + 2.*V*W*k5 + 2.*U*W*k6 + ((U*V)*lattice.cs2)*k7 + (-(2.*U*V)*lattice.cs2)*k8 + ((U*V)*lattice.cs2)*k9 + (V*0.5)*k10 + (U*0.5)*k11;
        double r23 = U*V2*W2*R + V2*W2*k1 + 2.*U*V*W2*k2 + 2.*U*V2*W*k3 + 2.*V*W2*k4 + 2.*V2*W*k5 + 4.*U*V*W*k6 + ((U*V2)*lattice.cs2 - (2.*U*W2)*lattice.cs2)*k7 + ((U*W2)*lattice.cs2 - (2.*U*V2)*lattice.cs2)*k8 + ((U*V2)*lattice.cs2 + (U*W2)*lattice.cs2)*k9 + (V2*0.5 + W2*0.5)*k10 + U*V*k11 + U*W*k12 + (U*0.5)*k17 + (-U*0.5)*k18 + k23;
        double r24 = U2*V*W2*R + 2.*U*V*W2*k1 + U2*W2*k2 + 2.*U2*V*W*k3 + 2.*U*W2*k4 + 4.*U*V*W*k5 + 2.*U2*W*k6 + ((V*U2)*lattice.cs2 + (V*W2)*lattice.cs2)*k7 + ((V*W2)*lattice.cs2 - (2.*U2*V)*lattice.cs2)*k8 + ((V*U2)*lattice.cs2 + (V*W2)*lattice.cs2)*k9 + U*V*k10 + (U2*0.5 + W2*0.5)*k11 + V*W*k12 + (V*0.25)*k17 + (V*0.25)*k18 + k24;
        double r25 = U2*V2*W*R + 2.*U*V2*W*k1 + 2.*U2*V*W*k2 + U2*V2*k3 + 4.*U*V*W*k4 + 2.*U*V2*k5 + 2.*U2*V*k6 + ((W*V2)*lattice.cs2 - (2.*U2*W)*lattice.cs2)*k7 + ((W*U2)*lattice.cs2 + (W*V2)*lattice.cs2)*k8 + ((W*U2)*lattice.cs2 + (W*V2)*lattice.cs2)*k9 + U*W*k10 + V*W*k11 + (U2*0.5 + V2*0.5)*k12 + (W*0.25)*k17 + (W*0.25)*k18 + k25;
        double r26 = U2*V2*W2*R + 2.*U*V2*W2*k1 + 2.*U2*V*W2*k2 + 2.*U2*V2*W*k3 + 4.*U*V*W2*k4 + 4.*U*V2*W*k5 + 4.*U2*V*W*k6 + ((U2*V2)*lattice.cs2 - (2.*U2*W2)*lattice.cs2 + (V2*W2)*lattice.cs2)*k7 + (- (2.*U2*V2)*lattice.cs2 + (U2*W2)*lattice.cs2 + (V2*W2)*lattice.cs2)*k8 + ((U2*V2)*lattice.cs2 + (U2*W2)*lattice.cs2 + (V2*W2)*lattice.cs2)*k9 + 
                    (U*V2 + U*W2)*k10 + (V*U2 + V*W2)*k11 + (W*U2 + W*V2)*k12 + (U2*0.5 + V2*0.25 + W2*0.25)*k17 + (- U2*0.5 + V2*0.25 + W2*0.25)*k18 + 2.*U*k23 + 2.*V*k24 + 2.*W*k25 + k26;
        
        f1(x, y, z, 0) = r0 - r9 + r17 - r26;
        f1(x, y, z, 1) = (r7 + r8 + r9)*0.5*lattice.cs2 - (r17 + r18)*0.25 - r10*0.5 + (r1 + r23 + r26)*0.5;
        f1(x, y, z, 2) = (r10 + r26)*0.5 - (r17 + r18)*0.25 - (r1 + r23)*0.5 + (r7 + r8 + r9)*0.5*lattice.cs2;
        f1(x, y, z, 3) = r18*0.125 - r11*0.5 - (3*r17)*0.125 - r7/3 - r19*0.25 + (r8 + r9)*0.5*lattice.cs2 + (r2 + r24 + r26)*0.5;
        f1(x, y, z, 4) = r18*0.125 - (3*r17)*0.125 - r7/3 - r19*0.25 + (r8 + r9)*0.5*lattice.cs2 - (r2 + r24)*0.5 + (r11 + r26)*0.5;
        f1(x, y, z, 5) = r18*0.125 - r12*0.5 - (3*r17)*0.125 - r8/3 + r19*0.25 + (r7 + r9)*0.5*lattice.cs2 + (r3 + r25 + r26)*0.5;
        f1(x, y, z, 6) = r18*0.125 - (3*r17)*0.125 - r8/3 + r19*0.25 + (r7 + r9)*0.5*lattice.cs2 - (r3 + r25)*0.5 + (r12 + r26)*0.5;
        f1(x, y, z, 7) = (r17 + r18)*0.0625 + r4*0.25 - (r22 + r23 + r24 + r26)*0.25 + (r10 + r11 + r13 + r14 + r19)*0.125;
        f1(x, y, z, 8) =  (r17 + r18)*0.0625 + (r22 + r23)*0.25 - (r10 + r13)*0.125 + (r11 + r14 + r19)*0.125 - (r4 + r24 + r26)*0.25;
        f1(x, y, z, 9) =  (r17 + r18)*0.0625 + (r22 + r24)*0.25 - (r11 + r14)*0.125 + (r10 + r13 + r19)*0.125 - (r4 + r23 + r26)*0.25;
        f1(x, y, z, 10) =  (r17 + r18)*0.0625 + r19*0.125 - (r22 + r26)*0.25 + (r4 + r23 + r24)*0.25 - (r10 + r11 + r13 + r14)*0.125;
        f1(x, y, z, 11) =  (r17 + r18)*0.0625 + r5*0.25 - (r13 + r19)*0.125 + (r10 + r12 + r15)*0.125 - (r21 + r23 + r25 + r26)*0.25;
        f1(x, y, z, 12) =  (r17 + r18)*0.0625 + (r21 + r23)*0.25 - (r10 + r19)*0.125 + (r12 + r13 + r15)*0.125 - (r5 + r25 + r26)*0.25;
        f1(x, y, z, 13) =  (r17 + r18)*0.0625 + r10*0.125 + (r21 + r25)*0.25 - (r5 + r23 + r26)*0.25 - (r12 + r13 + r15 + r19)*0.125;
        f1(x, y, z, 14) =  (r17 + r18)*0.0625 + r13*0.125 - (r21 + r26)*0.25 + (r5 + r23 + r25)*0.25 - (r10 + r12 + r15 + r19)*0.125;
        f1(x, y, z, 15) =  r6*0.25 + (r11 + r12 + r17)*0.125 - (r14 + r15 + r18)*0.125 - (r20 + r24 + r25 + r26)*0.25;
        f1(x, y, z, 16) =  (r20 + r24)*0.25 + (r12 + r14 + r17)*0.125 - (r11 + r15 + r18)*0.125 - (r6 + r25 + r26)*0.25;
        f1(x, y, z, 17) =  (r20 + r25)*0.25 + (r11 + r15 + r17)*0.125 - (r12 + r14 + r18)*0.125 - (r6 + r24 + r26)*0.25;
        f1(x, y, z, 18) =  (r14 + r15 + r17)*0.125 - (r11 + r12 + r18)*0.125 - (r20 + r26)*0.25 + (r6 + r24 + r25)*0.25;
        f1(x, y, z, 19) =  (r16 + r20 + r21 + r22 + r23 + r24 + r25 + r26)*0.125;
        f1(x, y, z, 20) =  (r20 + r24 + r25 + r26)*0.125 - (r16 + r21 + r22 + r23)*0.125;
        f1(x, y, z, 21) =  (r21 + r23 + r25 + r26)*0.125 - (r16 + r20 + r22 + r24)*0.125;
        f1(x, y, z, 22) =  (r16 + r22 + r25 + r26)*0.125 - (r20 + r21 + r23 + r24)*0.125;
        f1(x, y, z, 23) =  (r22 + r23 + r24 + r26)*0.125 - (r16 + r20 + r21 + r25)*0.125;
        f1(x, y, z, 24) =  (r16 + r21 + r24 + r26)*0.125 - (r20 + r22 + r23 + r25)*0.125;
        f1(x, y, z, 25) =  (r16 + r20 + r23 + r26)*0.125 - (r21 + r22 + r24 + r25)*0.125;
        f1(x, y, z, 26) =  (r20 + r21 + r22 + r26)*0.125 - (r16 + r23 + r24 + r25)*0.125;

        double KU = parameters.Kappa*EX+U;
        double KV = parameters.Kappa*EY+V;
        double KW = parameters.Kappa*EZ+W;
        double KU2 = KU*KU;
        double KV2 = KV*KV;
        double KW2 = KW*KW;
        r1 = h1(x, y, z, 1) - h1(x, y, z, 2) + h1(x, y, z, 7) - h1(x, y, z, 8) + h1(x, y, z, 9) - h1(x, y, z, 10) + h1(x, y, z, 11) - h1(x, y, z, 12) + h1(x, y, z, 13)- h1(x, y, z, 14) + h1(x, y, z, 19) - h1(x, y, z, 20) + h1(x, y, z, 21) - h1(x, y, z, 22) + h1(x, y, z, 23) - h1(x, y, z, 24) + h1(x, y, z, 25) - h1(x, y, z, 26);
        r2 = h1(x, y, z, 3) - h1(x, y, z, 4) + h1(x, y, z, 7) + h1(x, y, z, 8) - h1(x, y, z, 9) - h1(x, y, z, 10) + h1(x, y, z, 15) - h1(x, y, z, 16) + h1(x, y, z, 17) - h1(x, y, z, 18) + h1(x, y, z, 19) + h1(x, y, z, 20) - h1(x, y, z, 21) - h1(x, y, z, 22) + h1(x, y, z, 23) + h1(x, y, z, 24) - h1(x, y, z, 25) - h1(x, y, z, 26);
        r3 = h1(x, y, z, 5) - h1(x, y, z, 6) + h1(x, y, z, 11) + h1(x, y, z, 12) - h1(x, y, z, 13) - h1(x, y, z, 14) + h1(x, y, z, 15) + h1(x, y, z, 16) - h1(x, y, z, 17) - h1(x, y, z, 18) + h1(x, y, z, 19) + h1(x, y, z, 20) + h1(x, y, z, 21) + h1(x, y, z, 22) - h1(x, y, z, 23) - h1(x, y, z, 24) - h1(x, y, z, 25) - h1(x, y, z, 26);
        k1 = r1-KU*Q;
        k2 = r2-KV*Q;
        k3 = r3-KW*Q;
        k1 *= parameters.omega_q1;
        k2 *= parameters.omega_q1;
        k3 *= parameters.omega_q1;
        k9 = Q;
        k17 = Q*lattice.cs2;
        k18 = Q*lattice.cs2*lattice.cs2;
        k26 = Q*lattice.cs2*lattice.cs2*lattice.cs2;
        r1 = KU*Q + k1;
        r2 = KV*Q + k2;
        r3 = KW*Q + k3;
        r4 = KU*KV*Q + KV*k1 + KU*k2;
        r5 = KU*KW*Q + KW*k1 + KU*k3;
        r6 = KV*KW*Q + KW*k2 + KV*k3;
        r7 = (KU2 - KV2)*Q + 2.*KU*k1 - 2.*KV*k2;
        r8 = (KU2 - KW2)*Q + 2.*KU*k1 - 2.*KW*k3;
        r9 = (KU2 + KV2 + KW2)*Q + 2.*KU*k1 + 2.*KV*k2 + 2.*KW*k3 + k9;
        r10 = (KU*KV2 + KU*KW2)*Q + (KV2 + KW2)*k1 + 2.*KU*KV*k2 + 2.*KU*KW*k3 + 2.*KU*lattice.cs2*k9;
        r11 = (KV*KU2 + KV*KW2)*Q + 2.*KU*KV*k1 + (KU2 + KW2)*k2 + 2.*KV*KW*k3 + 2.*KV*lattice.cs2*k9;
        r12 = (KW*KU2 + KW*KV2)*Q + 2.*KU*KW*k1 + 2.*KV*KW*k2 + (KU2 + KV2)*k3 + 2.*KW*lattice.cs2*k9;
        r13 = (KU*KV2 - KU*KW2)*Q + (KV2 - KW2)*k1 + 2.*KU*KV*k2 - 2.*KU*KW*k3;
        r14 = (KV*KU2 - KV*KW2)*Q + 2.*KU*KV*k1 + (KU2 - KW2)*k2 - 2.*KV*KW*k3;
        r15 = (KW*KU2 - KW*KV2)*Q + 2.*KU*KW*k1 - 2.*KV*KW*k2 + (KU2 - KV2)*k3;
        r16 = KU*KV*KW*Q + KV*KW*k1 + KU*KW*k2 + KU*KV*k3;
        r17 = (KU2*KV2 + KU2*KW2 + KV2*KW2)*Q + 2.*KU*(KV2 + KW2)*k1 + 2.*KV*(KU2 + KW2)*k2 + 2.*KW*(KU2 + KV2)*k3 + 2.*lattice.cs2*(KU2 + KV2 + KW2)*k9 + k17;
        r18 = (KU2*KV2 + KU2*KW2 - KV2*KW2)*Q + 2.*KU*(KV2 + KW2)*k1 + 2.*KV*(KU2 - KW2)*k2 + 2.*KW*(KU2 - KV2)*k3 + 2.*KU2*lattice.cs2*k9 + k18;
        r19 = (KU2*KV2 - KU2*KW2)*Q + 2.*KU*(KV2 - KW2)*k1 + 2.*KU2*KV*k2 - 2*KU2*KW*k3 + (KV2 - KW2)*lattice.cs2*k9;
        r20 = KU2*KV*KW*Q + 2.*KU*KV*KW*k1 + KU2*KW*k2 + KU2*KV*k3 + KV*KW*lattice.cs2*k9;
        r21 = KU*KV2*KW*Q + KV2*KW*k1 + 2.*KU*KV*KW*k2 + KU*KV2*k3 + KU*KW*lattice.cs2*k9;
        r22 = KU*KV*KW2*Q + KV*KW2*k1 + KU*KW2*k2 + 2.*KU*KV*KW*k3 + KU*KV*lattice.cs2*k9;
        r23 = KU*KV2*KW2*Q + KV2*KW2*k1 + 2.*KU*KV*KW2*k2 + 2.*KU*KV2*KW*k3 + KU*(KV2 + KW2)*lattice.cs2*k9 + 0.5*KU*k17 - 0.5*KU*k18;
        r24 = KU2*KV*KW2*Q + 2.*KU*KV*KW2*k1 + KU2*KW2*k2 + 2.*KU2*KV*KW*k3 + KV*(KU2 + KW2)*lattice.cs2*k9 + 0.25*KV*k17 + 0.25*KV*k18;
        r25 = KU2*KV2*KW*Q + 2.*KU*KV2*KW*k1 + 2.*KU2*KV*KW*k2 + KU2*KV2*k3 + KW*(KU2 + KV2)*lattice.cs2*k9 + 0.25*KW*k17 + 0.25*KW*k18;
        r26 = KU2*KV2*KW2*Q + 2.*KU*KV2*KW2*k1 + 2.*KU2*KV*KW2*k2 + 2.*KU2*KV2*KW*k3 + (KU2*KV2 + KU2*KW2 + KV2*KW2)*lattice.cs2*k9 + (KU2*0.5 + KV2*0.25 + KW2*0.25)*k17 + (- KU2*0.5 + KV2*0.25 + KW2*0.25)*k18 + k26;

        h1(x, y, z, 0) = Q - r9 + r17 - r26;
        h1(x, y, z, 1) = (r7 + r8 + r9)*0.5*lattice.cs2 - (r17 + r18)*0.25 - r10*0.5 + (r1 + r23 + r26)*0.5;
        h1(x, y, z, 2) = (r10 + r26)*0.5 - (r17 + r18)*0.25 - (r1 + r23)*0.5 + (r7 + r8 + r9)*0.5*lattice.cs2;
        h1(x, y, z, 3) = r18*0.125 - r11*0.5 - (3*r17)*0.125 - r7/3 - r19*0.25 + (r8 + r9)*0.5*lattice.cs2 + (r2 + r24 + r26)*0.5;
        h1(x, y, z, 4) = r18*0.125 - (3*r17)*0.125 - r7/3 - r19*0.25 + (r8 + r9)*0.5*lattice.cs2 - (r2 + r24)*0.5 + (r11 + r26)*0.5;
        h1(x, y, z, 5) = r18*0.125 - r12*0.5 - (3*r17)*0.125 - r8/3 + r19*0.25 + (r7 + r9)*0.5*lattice.cs2 + (r3 + r25 + r26)*0.5;
        h1(x, y, z, 6) = r18*0.125 - (3*r17)*0.125 - r8/3 + r19*0.25 + (r7 + r9)*0.5*lattice.cs2 - (r3 + r25)*0.5 + (r12 + r26)*0.5;
        h1(x, y, z, 7) = (r17 + r18)*0.0625 + r4*0.25 - (r22 + r23 + r24 + r26)*0.25 + (r10 + r11 + r13 + r14 + r19)*0.125;
        h1(x, y, z, 8) =  (r17 + r18)*0.0625 + (r22 + r23)*0.25 - (r10 + r13)*0.125 + (r11 + r14 + r19)*0.125 - (r4 + r24 + r26)*0.25;
        h1(x, y, z, 9) =  (r17 + r18)*0.0625 + (r22 + r24)*0.25 - (r11 + r14)*0.125 + (r10 + r13 + r19)*0.125 - (r4 + r23 + r26)*0.25;
        h1(x, y, z, 10) =  (r17 + r18)*0.0625 + r19*0.125 - (r22 + r26)*0.25 + (r4 + r23 + r24)*0.25 - (r10 + r11 + r13 + r14)*0.125;
        h1(x, y, z, 11) =  (r17 + r18)*0.0625 + r5*0.25 - (r13 + r19)*0.125 + (r10 + r12 + r15)*0.125 - (r21 + r23 + r25 + r26)*0.25;
        h1(x, y, z, 12) =  (r17 + r18)*0.0625 + (r21 + r23)*0.25 - (r10 + r19)*0.125 + (r12 + r13 + r15)*0.125 - (r5 + r25 + r26)*0.25;
        h1(x, y, z, 13) =  (r17 + r18)*0.0625 + r10*0.125 + (r21 + r25)*0.25 - (r5 + r23 + r26)*0.25 - (r12 + r13 + r15 + r19)*0.125;
        h1(x, y, z, 14) =  (r17 + r18)*0.0625 + r13*0.125 - (r21 + r26)*0.25 + (r5 + r23 + r25)*0.25 - (r10 + r12 + r15 + r19)*0.125;
        h1(x, y, z, 15) =  r6*0.25 + (r11 + r12 + r17)*0.125 - (r14 + r15 + r18)*0.125 - (r20 + r24 + r25 + r26)*0.25;
        h1(x, y, z, 16) =  (r20 + r24)*0.25 + (r12 + r14 + r17)*0.125 - (r11 + r15 + r18)*0.125 - (r6 + r25 + r26)*0.25;
        h1(x, y, z, 17) =  (r20 + r25)*0.25 + (r11 + r15 + r17)*0.125 - (r12 + r14 + r18)*0.125 - (r6 + r24 + r26)*0.25;
        h1(x, y, z, 18) =  (r14 + r15 + r17)*0.125 - (r11 + r12 + r18)*0.125 - (r20 + r26)*0.25 + (r6 + r24 + r25)*0.25;
        h1(x, y, z, 19) =  (r16 + r20 + r21 + r22 + r23 + r24 + r25 + r26)*0.125;
        h1(x, y, z, 20) =  (r20 + r24 + r25 + r26)*0.125 - (r16 + r21 + r22 + r23)*0.125;
        h1(x, y, z, 21) =  (r21 + r23 + r25 + r26)*0.125 - (r16 + r20 + r22 + r24)*0.125;
        h1(x, y, z, 22) =  (r16 + r22 + r25 + r26)*0.125 - (r20 + r21 + r23 + r24)*0.125;
        h1(x, y, z, 23) =  (r22 + r23 + r24 + r26)*0.125 - (r16 + r20 + r21 + r25)*0.125;
        h1(x, y, z, 24) =  (r16 + r21 + r24 + r26)*0.125 - (r20 + r22 + r23 + r25)*0.125;
        h1(x, y, z, 25) =  (r16 + r20 + r23 + r26)*0.125 - (r21 + r22 + r24 + r25)*0.125;
        h1(x, y, z, 26) =  (r20 + r21 + r22 + r26)*0.125 - (r16 + r23 + r24 + r25)*0.125;

        double first_order;
        int newx, newy, newz;
        for(int k=0; k<lattice.np; k++) {
           // first_order = (parameters.Kappa*EX+U)*lattice.cx[k] + (parameters.Kappa*EY+V)*lattice.cy[k] + (parameters.Kappa*EZ+W)*lattice.cz[k];
           // h1(x, y, z, k) = parameters.omega_q1*h1(x, y, z, k) + parameters.omega_q * lattice.wf[k] * Q  * (1. + 3.*first_order + 4.5*pow(first_order,2) 
             //               - 1.5*(pow(parameters.Kappa*EX+U,2)+pow(parameters.Kappa*EY+V,2)+pow(parameters.Kappa*EZ+W,2)));
            g1(x, y, z, k) = parameters.omega_phi1*g1(x, y, z, k) + parameters.omega_phi*lattice.wf[k]*Phi + 
                                parameters.beta_phi*lattice.wf[k]*Q/parameters.permittivity + 0.5*parameters.beta_phi*lattice.wf[k]/parameters.permittivity*(Q-q(x, y, z));
            newx = x+lattice.cx[k];
            newy = y+lattice.cy[k];
            newz = z+lattice.cz[k];
            if(x==0 || x==parameters.nx-1)
              newx = (x+lattice.cx[k] + parameters.nx) % parameters.nx;
            if(y==0 || y==parameters.ny-1)
              newy = (y+lattice.cy[k] + parameters.ny) % parameters.ny;
            if(z==0 || z==parameters.nz-1)
              newz = (z+lattice.cz[k] + parameters.nz) % parameters.nz;
            f2(newx,newy,newz,k) = f1(x, y, z, k);
            h2(newx,newy,newz,k) = h1(x, y, z, k);
            g2(newx,newy,newz,k) = g1(x, y, z, k);
        }
        q(x, y, z) = Q;
    });
    Kokkos::fence(); 
}

void applyBoundary_X(View4DArray f1, View4DArray f2, View4DArray h1, View4DArray h2, View4DArray g1, View4DArray g2,
                           View3DArray rho, View3DArray u, View3DArray v, View3DArray w, View3DArray q, View3DArray phi, 
                            View3DArray Ex, View3DArray Ey, View3DArray Ez, const D3Q27 lattice, const Params parameters) {
    Kokkos::parallel_for("RegularisedX", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {parameters.nx, parameters.nz}),
    KOKKOS_LAMBDA(const int x, const int z) {
      int y = 0;  // Wall at x = 0
      double R, U, V, W, Pxx, Pxy, Pxz, Pyy, Pyz, Pzz, Hxx, Hxy, Hxz, Hyy, Hyz, Hzz, A, fneq, QPI, EX, EY, EZ;
      rho(x,y,z) = R = 1.;
      u(x,y,z) = U = 0.;
      v(x,y,z) = V = 0.;
      w(x,y,z) = W = 0.;
      q(x,y,z) = parameters.q0;
      phi(x,y,z) = parameters.phi_b;
      EX = Ex(x,y,z);
      EY = Ey(x,y,z);
      EZ = Ez(x,y,z);
      Pxx = Pxy = Pxz = Pyy = Pyz = Pzz = 0.;
      Hxx = Hxy = Hxz = Hyy = Hyz = Hzz = 0.;
      Kokkos::Array<double, lattice.np> Feq;
      Kokkos::Array<double, lattice.np> Heq;
      for(int k=0; k<lattice.np; k++)
      {
        A = U*lattice.cx[k] + V*lattice.cy[k] + W*lattice.cz[k];
        Feq[k] = lattice.wf[k]*R*(1.+3.*A+4.5*A*A-1.5*(U*U+V*V+W*W));
        fneq = f1(x,y,z,k)-Feq[k];
        Pxx += fneq*lattice.cx[k]*lattice.cx[k];
        Pxy += fneq*lattice.cx[k]*lattice.cy[k];
        Pxz += fneq*lattice.cx[k]*lattice.cz[k];
        Pyy += fneq*lattice.cy[k]*lattice.cy[k];
        Pyz += fneq*lattice.cy[k]*lattice.cz[k];
        Pzz += fneq*lattice.cz[k]*lattice.cz[k];

        A = (parameters.Kappa*EX+U)*lattice.cx[k]+(parameters.Kappa*EY+V)*lattice.cy[k]+(parameters.Kappa*EZ+W)*lattice.cz[k];
        Heq[k] = lattice.wf[k]*q(x,y,z)*(1.+3.*A+4.5*A*A-1.5*(pow(parameters.Kappa*EX+U,2)+pow(parameters.Kappa*EY+V,2)+pow(parameters.Kappa*EZ+W,2)));
        fneq = h1(x,y,z,k)-Heq[k];
        Hxx += fneq*lattice.cx[k]*lattice.cx[k];
        Hxy += fneq*lattice.cx[k]*lattice.cy[k];
        Hxz += fneq*lattice.cx[k]*lattice.cz[k];
        Hyy += fneq*lattice.cy[k]*lattice.cy[k];
        Hyz += fneq*lattice.cy[k]*lattice.cz[k];
        Hzz += fneq*lattice.cz[k]*lattice.cz[k];
      }
      for(int k=0; k<lattice.np; k++)
      {
        QPI = Pxx*(lattice.cx[k]*lattice.cx[k]-lattice.cs2)+2*Pxy*lattice.cx[k]*lattice.cy[k]+2*Pxz*lattice.cx[k]*lattice.cz[k]+
              Pyy*(lattice.cy[k]*lattice.cy[k]-lattice.cs2)+2*Pyz*lattice.cy[k]*lattice.cz[k]+Pzz*(lattice.cz[k]*lattice.cz[k]-lattice.cs2);
        f2(x,y,z,k) = Feq[k]+4.5*lattice.wf[k]*QPI;

        QPI = Hxx*(lattice.cx[k]*lattice.cx[k]-lattice.cs2)+2*Hxy*lattice.cx[k]*lattice.cy[k]+2*Hxz*lattice.cx[k]*lattice.cz[k]+
              Hyy*(lattice.cy[k]*lattice.cy[k]-lattice.cs2)+2*Hyz*lattice.cy[k]*lattice.cz[k]+Hzz*(lattice.cz[k]*lattice.cz[k]-lattice.cs2);
        h2(x,y,z,k) = Heq[k]+4.5*lattice.wf[k]*QPI;
        g2(x,y,z,k) = lattice.wf[k]*phi(x,y,z);
      }

      y = parameters.ny-1;
      rho(x,y,z) = R = 1.;
      u(x,y,z) = U = 0.;
      v(x,y,z) = V = 0.;
      w(x,y,z) = W = 0.;
      q(x,y,z) = 4./3.*q(x,y-1,z)-1./3.*q(x,y-2,z);
      phi(x,y,z) = parameters.phi_t;
      EX = Ex(x,y,z);
      EY = Ey(x,y,z);
      EZ = Ez(x,y,z);
      Pxx = Pxy = Pxz = Pyy = Pyz = Pzz = 0.;
      Hxx = Hxy = Hxz = Hyy = Hyz = Hzz = 0.;
      for(int k=0; k<lattice.np; k++)
      {
        A = U*lattice.cx[k] + V*lattice.cy[k] + W*lattice.cz[k];
        Feq[k] = lattice.wf[k]*R*(1.+3.*A+4.5*A*A-1.5*(U*U+V*V+W*W));
        fneq = f1(x,y,z,k)-Feq[k];
        Pxx += fneq*lattice.cx[k]*lattice.cx[k];
        Pxy += fneq*lattice.cx[k]*lattice.cy[k];
        Pxz += fneq*lattice.cx[k]*lattice.cz[k];
        Pyy += fneq*lattice.cy[k]*lattice.cy[k];
        Pyz += fneq*lattice.cy[k]*lattice.cz[k];
        Pzz += fneq*lattice.cz[k]*lattice.cz[k];

        A = (parameters.Kappa*EX+U)*lattice.cx[k]+(parameters.Kappa*EY+V)*lattice.cy[k]+(parameters.Kappa*EZ+W)*lattice.cz[k];
        Heq[k] = lattice.wf[k]*q(x,y,z)*(1.+3.*A+4.5*A*A-1.5*(pow(parameters.Kappa*EX+U,2)+pow(parameters.Kappa*EY+V,2)+pow(parameters.Kappa*EZ+W,2)));
        fneq = h1(x,y,z,k)-Heq[k];
        Hxx += fneq*lattice.cx[k]*lattice.cx[k];
        Hxy += fneq*lattice.cx[k]*lattice.cy[k];
        Hxz += fneq*lattice.cx[k]*lattice.cz[k];
        Hyy += fneq*lattice.cy[k]*lattice.cy[k];
        Hyz += fneq*lattice.cy[k]*lattice.cz[k];
        Hzz += fneq*lattice.cz[k]*lattice.cz[k];
      }
      for(int k=0; k<lattice.np; k++)
      {
        QPI = Pxx*(lattice.cx[k]*lattice.cx[k]-lattice.cs2)+2*Pxy*lattice.cx[k]*lattice.cy[k]+2*Pxz*lattice.cx[k]*lattice.cz[k]+
              Pyy*(lattice.cy[k]*lattice.cy[k]-lattice.cs2)+2*Pyz*lattice.cy[k]*lattice.cz[k]+Pzz*(lattice.cz[k]*lattice.cz[k]-lattice.cs2);
        f2(x,y,z,k) = Feq[k]+4.5*lattice.wf[k]*QPI;

        QPI = Hxx*(lattice.cx[k]*lattice.cx[k]-lattice.cs2)+2*Hxy*lattice.cx[k]*lattice.cy[k]+2*Hxz*lattice.cx[k]*lattice.cz[k]+
              Hyy*(lattice.cy[k]*lattice.cy[k]-lattice.cs2)+2*Hyz*lattice.cy[k]*lattice.cz[k]+Hzz*(lattice.cz[k]*lattice.cz[k]-lattice.cs2);
        h2(x,y,z,k) = Heq[k]+4.5*lattice.wf[k]*QPI;
        g2(x,y,z,k) = lattice.wf[k]*phi(x,y,z);
      }
    });
    Kokkos::fence();
}

void applyBoundary_Y(View4DArray f1, View4DArray f2, View4DArray h1, View4DArray h2,  View4DArray g1, View4DArray g2, 
                           View3DArray rho, View3DArray u, View3DArray v, View3DArray w, View3DArray q, View3DArray phi, 
                            View3DArray Ex, View3DArray Ey, View3DArray Ez, const D3Q27 lattice, const Params parameters) 
{
    Kokkos::parallel_for("RegularisedY", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 0}, {parameters.ny-1, parameters.nz}),
    KOKKOS_LAMBDA(const int y, const int z) {
      int x = 0;  // Wall at x = 0
      /*for(int k=0; k<lattice.np; k++)
      {
        if(lattice.cx[k]>0)
        {
            f2(x,y,z,k) = f1(x,y,z,lattice.opp[k]);
            h2(x,y,z,k) = h1(x,y,z,lattice.opp[k]);
        }
      }
      x = parameters.nx-1;
      for(int k=0; k<lattice.np; k++)
      {
        if(lattice.cx[k]<0)
        {
            f2(x,y,z,k) = f1(x,y,z,lattice.opp[k]);
            h2(x,y,z,k) = h1(x,y,z,lattice.opp[k]);
        }
      }
*/
      double R, U, V, W, Pxx, Pxy, Pxz, Pyy, Pyz, Pzz, Hxx, Hxy, Hxz, Hyy, Hyz, Hzz, A, fneq, QPI, EX, EY, EZ;
      rho(x,y,z) = R = rho(x+1,y,z);
      u(x,y,z) = U = 0.;
      v(x,y,z) = V = v(x+1,y,z);
      w(x,y,z) = W = w(x+1,y,z);
      q(x,y,z) = q(x+1,y,z);
      phi(x,y,z) = 4./3.*phi(1,y,z) - 1./3.*phi(2,y,z);
      EX = Ex(x,y,z);
      EY = Ey(x,y,z);
      EZ = Ez(x,y,z);
      Pxx = Pxy = Pxz = Pyy = Pyz = Pzz = 0.;
      Hxx = Hxy = Hxz = Hyy = Hyz = Hzz = 0.;
      Kokkos::Array<double, lattice.np> Feq;
      Kokkos::Array<double, lattice.np> Heq;
      for(int k=0; k<lattice.np; k++)
      {
        A = U*lattice.cx[k] + V*lattice.cy[k] + W*lattice.cz[k];
        Feq[k] = lattice.wf[k]*R*(1.+3.*A+4.5*A*A-1.5*(U*U+V*V+W*W));
        fneq = f1(x,y,z,k)-Feq[k];
        Pxx += fneq*lattice.cx[k]*lattice.cx[k];
        Pxy += fneq*lattice.cx[k]*lattice.cy[k];
        Pxz += fneq*lattice.cx[k]*lattice.cz[k];
        Pyy += fneq*lattice.cy[k]*lattice.cy[k];
        Pyz += fneq*lattice.cy[k]*lattice.cz[k];
        Pzz += fneq*lattice.cz[k]*lattice.cz[k];

        A = (parameters.Kappa*EX+U)*lattice.cx[k]+(parameters.Kappa*EY+V)*lattice.cy[k]+(parameters.Kappa*EZ+W)*lattice.cz[k];
        Heq[k] = lattice.wf[k]*q(x,y,z)*(1.+3.*A+4.5*A*A-1.5*(pow(parameters.Kappa*EX+U,2)+pow(parameters.Kappa*EY+V,2)+pow(parameters.Kappa*EZ+W,2)));
        fneq = h1(x,y,z,k)-Heq[k];
        Hxx += fneq*lattice.cx[k]*lattice.cx[k];
        Hxy += fneq*lattice.cx[k]*lattice.cy[k];
        Hxz += fneq*lattice.cx[k]*lattice.cz[k];
        Hyy += fneq*lattice.cy[k]*lattice.cy[k];
        Hyz += fneq*lattice.cy[k]*lattice.cz[k];
        Hzz += fneq*lattice.cz[k]*lattice.cz[k];
      }
      for(int k=0; k<lattice.np; k++)
      {
        QPI = Pxx*(lattice.cx[k]*lattice.cx[k]-lattice.cs2)+2*Pxy*lattice.cx[k]*lattice.cy[k]+2*Pxz*lattice.cx[k]*lattice.cz[k]+
              Pyy*(lattice.cy[k]*lattice.cy[k]-lattice.cs2)+2*Pyz*lattice.cy[k]*lattice.cz[k]+Pzz*(lattice.cz[k]*lattice.cz[k]-lattice.cs2);
        f2(x,y,z,k) = Feq[k]+4.5*lattice.wf[k]*QPI;

        QPI = Hxx*(lattice.cx[k]*lattice.cx[k]-lattice.cs2)+2*Hxy*lattice.cx[k]*lattice.cy[k]+2*Hxz*lattice.cx[k]*lattice.cz[k]+
              Hyy*(lattice.cy[k]*lattice.cy[k]-lattice.cs2)+2*Hyz*lattice.cy[k]*lattice.cz[k]+Hzz*(lattice.cz[k]*lattice.cz[k]-lattice.cs2);
        h2(x,y,z,k) = Heq[k]+4.5*lattice.wf[k]*QPI;
        g2(x,y,z,k) = lattice.wf[k]*phi(x,y,z);
      }

      x = parameters.nx-1;
      rho(x,y,z) = R = rho(x-1,y,z);
      u(x,y,z) = U = 0.;
      v(x,y,z) = V = v(x-1,y,z);
      w(x,y,z) = W = w(x-1,y,z);
      q(x,y,z) = q(x-1,y,z);
      phi(x,y,z) = 4./3.*phi(parameters.nx-2,y,z) - 1./3.*phi(parameters.nx-3,y,z);
      EX = Ex(x,y,z);
      EY = Ey(x,y,z);
      EZ = Ez(x,y,z);
      Pxx = Pxy = Pxz = Pyy = Pyz = Pzz = 0.;
      Hxx = Hxy = Hxz = Hyy = Hyz = Hzz = 0.;
      for(int k=0; k<lattice.np; k++)
      {
        A = U*lattice.cx[k] + V*lattice.cy[k] + W*lattice.cz[k];
        Feq[k] = lattice.wf[k]*R*(1.+3.*A+4.5*A*A-1.5*(U*U+V*V+W*W));
        fneq = f1(x,y,z,k)-Feq[k];
        Pxx += fneq*lattice.cx[k]*lattice.cx[k];
        Pxy += fneq*lattice.cx[k]*lattice.cy[k];
        Pxz += fneq*lattice.cx[k]*lattice.cz[k];
        Pyy += fneq*lattice.cy[k]*lattice.cy[k];
        Pyz += fneq*lattice.cy[k]*lattice.cz[k];
        Pzz += fneq*lattice.cz[k]*lattice.cz[k];

        A = (parameters.Kappa*EX+U)*lattice.cx[k]+(parameters.Kappa*EY+V)*lattice.cy[k]+(parameters.Kappa*EZ+W)*lattice.cz[k];
        Heq[k] = lattice.wf[k]*q(x,y,z)*(1.+3.*A+4.5*A*A-1.5*(pow(parameters.Kappa*EX+U,2)+pow(parameters.Kappa*EY+V,2)+pow(parameters.Kappa*EZ+W,2)));
        fneq = h1(x,y,z,k)-Heq[k];
        Hxx += fneq*lattice.cx[k]*lattice.cx[k];
        Hxy += fneq*lattice.cx[k]*lattice.cy[k];
        Hxz += fneq*lattice.cx[k]*lattice.cz[k];
        Hyy += fneq*lattice.cy[k]*lattice.cy[k];
        Hyz += fneq*lattice.cy[k]*lattice.cz[k];
        Hzz += fneq*lattice.cz[k]*lattice.cz[k];
      }
      for(int k=0; k<lattice.np; k++)
      {
        QPI = Pxx*(lattice.cx[k]*lattice.cx[k]-lattice.cs2)+2*Pxy*lattice.cx[k]*lattice.cy[k]+2*Pxz*lattice.cx[k]*lattice.cz[k]+
              Pyy*(lattice.cy[k]*lattice.cy[k]-lattice.cs2)+2*Pyz*lattice.cy[k]*lattice.cz[k]+Pzz*(lattice.cz[k]*lattice.cz[k]-lattice.cs2);
        f2(x,y,z,k) = Feq[k]+4.5*lattice.wf[k]*QPI;

        QPI = Hxx*(lattice.cx[k]*lattice.cx[k]-lattice.cs2)+2*Hxy*lattice.cx[k]*lattice.cy[k]+2*Hxz*lattice.cx[k]*lattice.cz[k]+
              Hyy*(lattice.cy[k]*lattice.cy[k]-lattice.cs2)+2*Hyz*lattice.cy[k]*lattice.cz[k]+Hzz*(lattice.cz[k]*lattice.cz[k]-lattice.cs2);
        h2(x,y,z,k) = Heq[k]+4.5*lattice.wf[k]*QPI;
        g2(x,y,z,k) = lattice.wf[k]*phi(x,y,z);
      }
    });
    Kokkos::fence();
}

void swap_pop(View4DArray f1, View4DArray f2, View4DArray h1, View4DArray h2, View4DArray g1, View4DArray g2, const D3Q27 lattice, const Params parameters) {

    // Kokkos lambda for initialization
    Kokkos::parallel_for("swap_pop",
    Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0}, {parameters.nx, parameters.ny, parameters.nz, lattice.np}),
    KOKKOS_LAMBDA(const int x, const int y, const int z, const int k) {

        f1(x, y, z, k) = f2(x, y, z, k);
        h1(x, y, z, k) = h2(x, y, z, k);
        g1(x, y, z, k) = g2(x, y, z, k);
    }
    );
    Kokkos::fence(); 
}

int main(int argc, char *argv[]) {
    const D3Q27 lattice;
    const Params parameters;
    // Define the dimensions and views
    namespace fs = std::filesystem;
    fs::path dirname = "./EHD_3D_Q27";
    bool first_write = true;  // Ensures file is cleared only once


    for ( int i = 0; i < argc; i++ ) {
        if (  strcmp( argv[ i ], "-d" ) == 0 ) {
            dirname = argv[ ++i ];
            printf( "Using output directory, %d\n", parameters.nx );
        }
    }
    // Open the output file
    if (fs::exists(dirname)) {
        std::cout << "Directory exists. Overwriting..." << dirname << std::endl;
    }
    else {
        fs::create_directories(dirname);
    }

    

    fs::path output_path;
    fs::path data_path = dirname / "dataQ27.txt";
    int check_mach = 0;
    int check = 0;
    double Jmax, Vortmax;
    double kinetic_enstrophy, kinetic_energy, magnetic_energy,
        rho_average, magnetic_enstrophy;

    // Profiling parameters
    double time_init, time_algo;

    Kokkos::initialize(argc, argv); {
        // Create Kokkos Views for fluid arrays
        View4DArray f1("f1", parameters.nx, parameters.ny, parameters.nz, lattice.np);
        View4DArray f2("f2", parameters.nx, parameters.ny, parameters.nz, lattice.np);
        View3DArray rho("rho", parameters.nx, parameters.ny, parameters.nz);
        View3DArray u("u", parameters.nx, parameters.ny, parameters.nz);
        View3DArray v("v", parameters.nx, parameters.ny, parameters.nz);
        View3DArray w("w", parameters.nx, parameters.ny, parameters.nz);

        View4DArray h1("h1", parameters.nx, parameters.ny, parameters.nz, lattice.np);
        View4DArray h2("h2", parameters.nx, parameters.ny, parameters.nz, lattice.np);
        View3DArray q("q", parameters.nx, parameters.ny, parameters.nz);
        View3DArray Ex("Ex", parameters.nx, parameters.ny, parameters.nz);
        View3DArray Ey("Ey", parameters.nx, parameters.ny, parameters.nz);
        View3DArray Ez("Ez", parameters.nx, parameters.ny, parameters.nz);
        View3DArray phi("phi", parameters.nx, parameters.ny, parameters.nz);
        View3DArray phi_old("phi_old", parameters.nx, parameters.ny, parameters.nz);
        View4DArray g1("g1", parameters.nx, parameters.ny, parameters.nz, lattice.np);
        View4DArray g2("g2", parameters.nx, parameters.ny, parameters.nz, lattice.np);

        auto h_f1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f1);
        auto h_f2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f2);
        auto h_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u);
        auto h_v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
        auto h_w = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), w);
        auto h_rho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rho);

        auto h_h1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), h1);
        auto h_h2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), h2);
        auto h_q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), q);
        auto h_Ex = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Ex);
        auto h_Ey = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Ey);
        auto h_Ez = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Ez);
        auto h_phi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), phi);
        auto h_phi_old = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), phi_old);
        auto h_g1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), g1);
        auto h_g2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), g2);

        // Call the Kokkos-enabled initialization function
        {
            Kokkos::Timer timer;
            initial_state(rho, u, v, w, f1, f2, h1, h2, g1, g2, q, phi, Ex, Ey, Ez, lattice, parameters);
            Kokkos::deep_copy(h_rho, rho);
                    Kokkos::deep_copy(h_u, u);
                    Kokkos::deep_copy(h_v, v);
                    Kokkos::deep_copy(h_w, w);
                    Kokkos::deep_copy(h_Ex, Ex);
                    Kokkos::deep_copy(h_Ey, Ey);
                    Kokkos::deep_copy(h_Ez, Ez);
                    Kokkos::deep_copy(h_q, q);
                    Kokkos::deep_copy(h_phi, phi);
                    //output_path = dirname / ("fluid_t_-1.vtk");
                    //VTKWriter::write(output_path.string(), parameters.nx, parameters.ny, parameters.nz, h_u, h_v, h_w,
                      //               h_rho, h_Ex, h_Ey, h_Ez, h_q, h_phi, parameters.q0, parameters.phi_b, parameters.U_ref);

            Kokkos::fence();
            time_init = timer.seconds();
            Kokkos::deep_copy(h_f1, f1);
            Kokkos::deep_copy(h_f2, f2);
            Kokkos::deep_copy(h_h1, h1);
            Kokkos::deep_copy(h_h2, h2);
            Kokkos::deep_copy(h_g1, g1);
            Kokkos::deep_copy(h_g2, g2);
        }
        
        {
            Kokkos::Timer timer;
            int l;
            for(int i=0; i<parameters.nsteps; i++) {
                copy_phi(phi, phi_old, parameters);
                algoLB(f1, f2, h1, h2, g1, g2, rho, u, v, w, Ex, Ey, Ez, q, phi, phi_old, lattice, parameters);
                applyBoundary_X(f1, f2, h1, h2, g1, g2, rho, u, v, w, q, phi, Ex, Ey, Ez, lattice, parameters);
                applyBoundary_Y(f1, f2, h1, h2, g1, g2, rho, u, v, w, q, phi, Ex, Ey, Ez, lattice, parameters);
                swap_pop(f1, f2, h1, h2, g1, g2, lattice, parameters); 
                //std::swap(f1, f2);
                //std::swap(h1, h2);
                if(i%parameters.n_out == 0){
                    writeData(data_path, i, parameters.T_ref, v(parameters.nx/2, parameters.ny/2, parameters.nz/2)/parameters.U_ref, first_write);
                   first_write = false;
                }
                if(i%parameters.n_out == 0)
                printf("t %lf of %lf. A = %lf\n", i/parameters.T_ref, parameters.nsteps/parameters.T_ref, v(parameters.nx/2, parameters.ny/2, parameters.nz/2)/parameters.U_ref);

                if(plot_vtk==true && i%parameters.n_out==0){
                    output_path = dirname / ("fluid_t_" + std::to_string(i) + ".vtk");
                    // Copy data from device to host
                    Kokkos::deep_copy(h_rho, rho);
                    Kokkos::deep_copy(h_u, u);
                    Kokkos::deep_copy(h_v, v);
                    Kokkos::deep_copy(h_w, w);
                    Kokkos::deep_copy(h_Ex, Ex);
                    Kokkos::deep_copy(h_Ey, Ey);
                    Kokkos::deep_copy(h_Ez, Ez);
                    Kokkos::deep_copy(h_q, q);
                    Kokkos::deep_copy(h_phi, phi);
                    VTKWriter::write(output_path.string(), parameters.nx, parameters.ny, parameters.nz, h_u, h_v, h_w,
                                     h_rho, h_Ex, h_Ey, h_Ez, h_q, h_phi, parameters.q0, parameters.phi_b, parameters.U_ref);
                }

                // check the Mach number...if too high, it exits!
                if(check_mach == 1) goto labelA;
            }
            Kokkos::fence();
            time_algo = timer.seconds();
        }
    }
    labelA:
    Kokkos::finalize();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Timing Results:" << std::endl;
    std::cout << "initialization time: " << time_init << " s" << std::endl;
    std::cout << "algorithm time: " << time_algo << " s" << std::endl;
    return 0;
}