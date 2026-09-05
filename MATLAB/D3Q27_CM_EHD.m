clear all
clc
%addpath('/Applications/Matlab_R2019b.app/toolbox/symbolic/symbolic')
syms U V W R Fx Fy Fz omega k4_star k5_star k6_star k7_star k8_star...
    f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16 f17 f18 f19 f20 f21 f22 f23 f24 f25 f26 real
f = [f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16 f17 f18 f19 f20 f21 f22 f23 f24 f25 f26];
L = diag([1, 1, 1, 1, omega, omega, omega, omega, omega, 1, 1, 1, 1, 1,...
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
cs = 1/sqrt(3);
cs2 = cs^2;
cs3 = cs^3;
cs4 = cs^4;
cs5 = cs^5;
cs6 = cs^6;
cs8 = cs^8;
cs10 = cs^10;
cs12 = cs^12;
T = sym(zeros(27,27));
M = zeros(27,27);
feq = sym(zeros(27,1));
cx = [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1];
cy = [0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1];
cz = [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1];
w = [8./27., 2./27., 2./27., 2./27., 2./27., 2./27., 2./27.,... 
	1./54., 1./54., 1./54., 1./54., 1./54., 1./54., 1./54., 1./54., 1./54., 1./54., 1./54., 1./54.,...
	1./216., 1./216., 1./216., 1./216., 1./216., 1./216., 1./216., 1./216.];
u2 = U*U+V*V+W*W;
%choose the maximum order of the Hermite polynomials N=2,3,4,5,6
for i=1:length(cx)
    first_order = U*cx(i)+V*cy(i)+W*cz(i);
    first_order = first_order/cs2;
    
    second_order = 4.5*(U*cx(i)+V*cy(i)+W*cz(i))^2-1.5*u2;
    
    third_order = (cx(i)^2-cs2)*cy(i)*U*U*V + (cx(i)^2-cs2)*cz(i)*U*U*W +...
                  (cy(i)^2-cs2)*cx(i)*U*V*V + (cz(i)^2-cs2)*cx(i)*U*W*W +...
                  (cz(i)^2-cs2)*cy(i)*V*W*W + (cy(i)^2-cs2)*cz(i)*V*V*W +...
              2*( cx(i)*cy(i)*cz(i)*U*V*W);  
    third_order = third_order/(2*cs6);

    fourth_order = (cx(i)^2-cs2)*(cy(i)^2-cs2)*U*U*V*V +...
                   (cx(i)^2-cs2)*(cz(i)^2-cs2)*U*U*W*W +...
                   (cy(i)^2-cs2)*(cz(i)^2-cs2)*V*V*W*W +...
                2*( cx(i)*cy(i)*(cz(i)^2-cs2)*U*V*W*W +...
                    cx(i)*(cy(i)^2-cs2)*cz(i)*U*V*V*W +...
                   (cx(i)^2-cs2)*cy(i)*cz(i)*U*U*V*W );
    fourth_order = fourth_order/(4*cs8);
    
    fifth_order = (cx(i)^2-cs2)*cy(i)*(cz(i)^2-cs2)*U*U*V*W*W +...
                  (cx(i)^2-cs2)*(cy(i)^2-cs2)*cz(i)*U*U*V*V*W +...
                  cx(i)*(cy(i)^2-cs2)*(cz(i)^2-cs2)*U*V*V*W*W;
    fifth_order = fifth_order/(4*cs10);

    sixth_order = (cx(i)^2-cs2)*(cy(i)^2-cs2)*(cz(i)^2-cs2)*U*U*V*V*W*W;
    sixth_order = sixth_order/(8*cs12);
    feq(i) = R*w(i)*(1+first_order + second_order + third_order +...
                         fourth_order + fifth_order + sixth_order);
    %A = U*cx(i)+V*cy(i)+W*cz(i);
    %force(i) = w(i)*( Fx*(3*(cx(i)-U)+9*A*cx(i))+...
     %                     Fy*(3*(cy(i)-V)+9*A*cy(i))+...
	%					  Fz*(3*(cz(i)-W)+9*A*cz(i)) );


    % Forcing term
    hat_cx = cx(i)/cs;
    hat_cy = cy(i)/cs;
    hat_cz = cz(i)/cs;
    hat_ = [hat_cx, hat_cy, hat_cz];
    H1 = zeros(3,1);
    H2 = zeros(3,3);
    H3 = zeros(3,3,3);
    H4 = zeros(3,3,3,3);
    H5 = zeros(3,3,3,3,3);
    H6 = zeros(3,3,3,3,3,3);
    H1(1) = hat_cx;
    H1(2) = hat_cy;
    H1(3) = hat_cz;
    H2(1,1) = hat_cx^2-1;
    H2(1,2) = hat_cx*hat_cy;
    H2(1,3) = hat_cx*hat_cz;
    H2(2,1) = hat_cy*hat_cx;
    H2(2,2) = hat_cy^2-1;
    H2(2,3) = hat_cy*hat_cz;
    H2(3,1) = hat_cz*hat_cx;
    H2(3,2) = hat_cz*hat_cy;
    H2(3,3) = hat_cz^2-1;  
    H3(1,1,2) = (hat_cx^2-1)*hat_cy;
    H3(1,1,3) = (hat_cx^2-1)*hat_cz;
    H3(1,2,2) = hat_cx*(hat_cy^2-1);
    H3(1,3,3) = hat_cx*(hat_cz^2-1);
    H3(2,3,3) = hat_cy*(hat_cz^2-1);
    H3(2,2,3) = (hat_cy^2-1)*hat_cz;
    H3(1,2,3) = hat_cx*hat_cy*hat_cz;
    H3(2,3,1) = H3(1,2,3);
    H3(3,1,2) = H3(1,2,3);
    H4(1,1,2,2) = (hat_cx^2-1)*(hat_cy^2-1);
    H4(1,1,3,3) = (hat_cx^2-1)*(hat_cz^2-1);
    H4(2,2,3,3) = (hat_cy^2-1)*(hat_cz^2-1);
    H4(1,2,3,3) = hat_cx*hat_cy*(hat_cz^2-1);
    H4(1,2,2,3) = hat_cx*(hat_cy^2-1)*hat_cz;
    H4(1,1,2,3) = (hat_cx^2-1)*hat_cy*hat_cz;
    H5(1,1,2,3,3) = (hat_cx^2-1)*hat_cy*(hat_cz^2-1);
    H5(1,1,2,2,3) = (hat_cx^2-1)*(hat_cy^2-1)*hat_cz;
    H5(1,2,2,3,3) = hat_cx*(hat_cy^2-1)*(hat_cz^2-1);
    H6(1,1,2,2,3,3) = (hat_cx^2-1)*(hat_cy^2-1)*(hat_cz^2-1);
    first_order = (Fx*H1(1)+Fy*H1(2)+Fz*H1(3))/cs;
    second_order = 1/(2*cs2)*(H2(1,1)*(Fx*U+...
                                       U*Fx)+...
                              H2(2,2)*(Fy*V+...
                                       V*Fy)+... 
                              H2(3,3)*(Fz*W+...
                                       W*Fz)+...                                    
                            2*H2(1,2)*(Fx*V+...
                                       U*Fy)+...
                            2*H2(1,3)*(Fx*W+...
                                       U*Fz)+...
                            2*H2(2,3)*(Fy*W+...
                                       V*Fz) );
    third_order =  1/(6*cs3)*(3*H3(1,1,2)*(Fx*U*V+...
                                           U*Fx*V+...
                                           U*U*Fy)+...
                              3*H3(1,1,3)*(Fx*U*W+...
                                           U*Fx*W+...
                                           U*U*Fz)+...                         
                              3*H3(1,2,2)*(Fx*V*V+...
                                           U*Fy*V+...
                                           U*V*Fy)+... 
                              3*H3(1,3,3)*(Fx*W*W+...
                                           U*Fz*W+...
                                           U*W*Fz)+... 
                              3*H3(2,3,3)*(Fy*W*W+...
                                           V*Fz*W+...
                                           V*W*Fz)+... 
                              3*H3(2,2,3)*(Fy*V*W+...
                                           V*Fy*W+...
                                           V*V*Fz)+...
                              6*H3(1,2,3)*(Fx*V*W+...
                                           U*Fy*W+...
                                           U*V*Fz) );
    fourth_order = 1./(24*cs4)*(6*H4(1,1,2,2)*(Fx*U*V*V+...
                                               U*Fx*V*V+...
                                               U*U*Fy*V+...
                                               U*U*V*Fy)+...
                                6*H4(1,1,3,3)*(Fx*U*W*W+...
                                               U*Fx*W*W+...
                                               U*U*Fz*W+...
                                               U*U*W*Fz)+...
                                6*H4(2,2,3,3)*(Fy*V*W*W+...
                                               V*Fy*W*W+...
                                               V*V*Fz*W+...
                                               V*V*W*Fz)+...     
                               12*H4(1,2,3,3)*(Fx*V*W*W+...
                                               U*Fy*W*W+...
                                               U*V*Fz*W+...
                                               U*V*W*Fz)+...
                               12*H4(1,2,2,3)*(Fx*V*V*W+...
                                               U*Fy*V*W+...
                                               U*V*Fy*W+...
                                               U*V*V*Fz)+...
                               12*H4(1,1,2,3)*(Fx*U*V*W+...
                                               U*Fx*V*W+...
                                               U*U*Fy*W+...
                                               U*U*V*Fz) );
    fifth_order = 1./(120*cs5)*(30*H5(1,1,2,3,3)*(Fx*U*V*W*W+...
                                                  U*Fx*V*W*W+...
                                                  U*U*Fy*W*W+...
                                                  U*U*V*Fz*W+...
                                                  U*U*V*W*Fz)+...
                                30*H5(1,1,2,2,3)*(Fx*U*V*V*W+...
                                                  U*Fx*V*V*W+...
                                                  U*U*Fy*V*W+...
                                                  U*U*V*Fy*W+...
                                                  U*U*V*V*Fz)+...
                                30*H5(1,2,2,3,3)*(Fx*V*V*W*W+...
                                                  U*Fy*V*W*W+...
                                                  U*V*Fy*W*W+...
                                                  U*V*V*Fz*W+...
                                                  U*V*V*W*Fz) );
    sixth_order = 1./(720*cs6)*90*H6(1,1,2,2,3,3)*(Fx*U*V*V*W*W+...
                                                   U*Fx*V*V*W*W+...
                                                   U*U*Fy*V*W*W+...
                                                   U*U*V*Fy*W*W+...
                                                   U*U*V*V*Fz*W+...
                                                   U*U*V*V*W*Fz);

    force(i) = w(i)*(first_order + second_order + third_order +...
                         fourth_order + fifth_order + sixth_order);
    % Set the basis
    CX = cx(i)-U;
    CY = cy(i)-V;
    CZ = cz(i)-W;
    T(1,i) = 1;
    T(2,i) = CX;
    T(3,i) = CY;
    T(4,i) = CZ;
    T(5,i) = CX*CY;
    T(6,i) = CX*CZ;
    T(7,i) = CY*CZ;
    T(8,i) = CX*CX-CY*CY;
    T(9,i) = CX*CX-CZ*CZ;
    T(10,i) = CX*CX+CY*CY+CZ*CZ;
    T(11,i) = CX*CY*CY+CX*CZ*CZ;
    T(12,i) = CX*CX*CY+CY*CZ*CZ;
    T(13,i) = CX*CX*CZ+CY*CY*CZ;
    T(14,i) = CX*CY*CY-CX*CZ*CZ;
    T(15,i) = CX*CX*CY-CY*CZ*CZ;
    T(16,i) = CX*CX*CZ-CY*CY*CZ;
    T(17,i) = CX*CY*CZ;
    T(18,i) = CX*CX*CY*CY+CX*CX*CZ*CZ+CY*CY*CZ*CZ;
    T(19,i) = CX*CX*CY*CY+CX*CX*CZ*CZ-CY*CY*CZ*CZ;
    T(20,i) = CX*CX*CY*CY-CX*CX*CZ*CZ;
    T(21,i) = CX*CX*CY*CZ;
    T(22,i) = CX*CY*CY*CZ;
    T(23,i) = CX*CY*CZ*CZ;
    T(24,i) = CX*CY*CY*CZ*CZ;
    T(25,i) = CX*CX*CY*CZ*CZ;
    T(26,i) = CX*CX*CY*CY*CZ;
    T(27,i) = CX*CX*CY*CY*CZ*CZ;
    
    CX = cx(i);
    CY = cy(i);
    CZ = cz(i);
    M(1,i) = 1;
    M(2,i) = CX;
    M(3,i) = CY;
    M(4,i) = CZ;
    M(5,i) = CX*CY;
    M(6,i) = CX*CZ;
    M(7,i) = CY*CZ;
    M(8,i) = CX*CX-CY*CY;
    M(9,i) = CX*CX-CZ*CZ;
    M(10,i) = CX*CX+CY*CY+CZ*CZ;
    M(11,i) = CX*CY*CY+CX*CZ*CZ;
    M(12,i) = CX*CX*CY+CY*CZ*CZ;
    M(13,i) = CX*CX*CZ+CY*CY*CZ;
    M(14,i) = CX*CY*CY-CX*CZ*CZ;
    M(15,i) = CX*CX*CY-CY*CZ*CZ;
    M(16,i) = CX*CX*CZ-CY*CY*CZ;
    M(17,i) = CX*CY*CZ;
    M(18,i) = CX*CX*CY*CY+CX*CX*CZ*CZ+CY*CY*CZ*CZ;
    M(19,i) = CX*CX*CY*CY+CX*CX*CZ*CZ-CY*CY*CZ*CZ;
    M(20,i) = CX*CX*CY*CY-CX*CX*CZ*CZ;
    M(21,i) = CX*CX*CY*CZ;
    M(22,i) = CX*CY*CY*CZ;
    M(23,i) = CX*CY*CZ*CZ;
    M(24,i) = CX*CY*CY*CZ*CZ;
    M(25,i) = CX*CX*CY*CZ*CZ;
    M(26,i) = CX*CX*CY*CY*CZ;
    M(27,i) = CX*CX*CY*CY*CZ*CZ;
end
% Compute central moments
T = simplify(T);
N = simplify(T*M^(-1)); %shift matrix
%T = M;
%N = eye(27,27);
% N = [                          1,                                                                                                                                         0,                0,                0,         0,         0,         0,                                         0,                                           0,                                       0,              0,              0,              0,             0,             0,             0,        0,                     0,                       0,             0,     0,     0,     0,    0,    0,    0, 0;...
%                          -U,                                                                                                                                     1 ,                0,                0,         0,         0,         0,                                         0,                                           0,                                       0,              0,              0,              0,             0,             0,             0,        0,                     0,                       0,             0,     0,     0,     0,    0,    0,    0, 0;...
%                          -V,                                                                                                                                        0,                1,                0,         0,         0,         0,                                         0,                                           0,                                       0,              0,              0,              0,             0,             0,             0,        0,                     0,                       0,             0,     0,     0,     0,    0,    0,    0, 0;...
%                          -W,                                                                                                                                        0,                0,                1,         0,         0,         0,                                         0,                                           0,                                       0,              0,              0,              0,             0,             0,             0,        0,                     0,                       0,             0,     0,     0,     0,    0,    0,    0, 0;...
%                         U*V,                                                                                                               -V,               -U,                0,         1,         0,         0,                                         0,                                           0,                                       0,              0,              0,              0,             0,             0,             0,        0,                     0,                       0,             0,     0,     0,     0,    0,    0,    0, 0;...
%                         U*W,                                                                                                               -W,                0,               -U,         0,         1,         0,                                         0,                                           0,                                       0,              0,              0,              0,             0,             0,             0,        0,                     0,                       0,             0,     0,     0,     0,    0,    0,    0, 0;...
%                         V*W,                                                                                                                                     0,               -W,               -V,         0,         0,         1,                                         0,                                           0,                                       0,              0,              0,              0,             0,             0,             0,        0,                     0,                       0,             0,     0,     0,     0,    0,    0,    0, 0;...
%                   U^2 - V^2,                                                                                   - 2*U ,              2*V,                0,         0,         0,         0,                                         1,                                           0,                                       0,              0,              0,              0,             0,             0,             0,        0,                     0,                       0,             0,     0,     0,     0,    0,    0,    0, 0;...
%                   U^2 - W^2,                                                                                                         - 2*U,                0,              2*W,         0,         0,         0,                                         0,                                           1,                                       0,              0,              0,              0,             0,             0,             0,        0,                     0,                       0,             0,     0,     0,     0,    0,    0,    0, 0;...
%             U^2 + V^2 + W^2,                                                           - 2*U ,             -2*V,             -2*W,         0,         0,         0,                                         0,                                           0,                                       1,              0,              0,              0,             0,             0,             0,        0,                     0,                       0,             0,     0,     0,     0,    0,    0,    0, 0;...
%              -U*(V^2 + W^2),                                                                     V^2  + W^2,            2*U*V,            2*U*W,      -2*V,      -2*W,         0,                                       U/3,                                         U/3,                                -(2*U)/3,              1,              0,              0,             0,             0,             0,        0,                     0,                       0,             0,     0,     0,     0,    0,    0,    0, 0;...
%              -V*(U^2 + W^2),                                                                                                    2*U*V,        U^2 + W^2,            2*V*W,      -2*U,         0,      -2*W,                                  -(2*V)/3,                                         V/3,                                -(2*V)/3,              0,              1,              0,             0,             0,             0,        0,                     0,                       0,             0,     0,     0,     0,    0,    0,    0, 0;...
%              -W*(U^2 + V^2),                                                                                                - 2*W*U,            2*V*W,        U^2 + V^2,         0,      -2*U,      -2*V,                                       W/3,                                    -(2*W)/3,                                -(2*W)/3,              0,              0,              1,             0,             0,             0,        0,                     0,                       0,             0,     0,     0,     0,    0,    0,    0, 0;...
%              -U*(V^2 - W^2),                                                                     + V^2 - W^2,            2*U*V,           -2*U*W,      -2*V,       2*W,         0,                                         U,                                          -U,                                       0,              0,              0,              0,             1,             0,             0,        0,                     0,                       0,             0,     0,     0,     0,    0,    0,    0, 0;...
%              -V*(U^2 - W^2),                                                                                                   2*U*V,        U^2 - W^2,           -2*V*W,      -2*U,         0,       2*W,                                         0,                                          -V,                                       0,              0,              0,              0,             0,             1,             0,        0,                     0,                       0,             0,     0,     0,     0,    0,    0,    0, 0;...
%              -W*(U^2 - V^2),                                                                                               2*U*W,           -2*V*W,        U^2 - V^2,         0,      -2*U,       2*V,                                        -W,                                           0,                                       0,              0,              0,              0,             0,             0,             1,        0,                     0,                       0,             0,     0,     0,     0,    0,    0,    0, 0;...
%                      -U*V*W,                                                                                                            V*W,              U*W,              U*V,        -W,        -V,        -U,                                         0,                                           0,                                       0,              0,              0,              0,             0,             0,             0,        1,                     0,                       0,             0,     0,     0,     0,    0,    0,    0, 0;...
% U^2*V^2 + U^2*W^2 + V^2*W^2, - 2*U*V^2 - 2*U*W^2, -2*V*(U^2 + W^2), -2*W*(U^2 + V^2),     4*U*V,     4*U*W,     4*V*W,               - U^2/3 + (2*V^2)/3 - W^2/3,                 - U^2/3 - V^2/3 + (2*W^2)/3,       (2*U^2)/3 + (2*V^2)/3 + (2*W^2)/3,           -2*U,           -2*V,           -2*W,             0,             0,             0,        0,                     1,                       0,             0,     0,     0,     0,    0,    0,    0, 0;...
% U^2*V^2 + U^2*W^2 - V^2*W^2, - 2*U*V^2 - 2*U*W^2, -2*V*(U^2 - W^2), -2*W*(U^2 - V^2),     4*U*V,     4*U*W,    -4*V*W,                               W^2 - U^2/3,                                 V^2 - U^2/3,                               (2*U^2)/3,           -2*U,              0,              0,             0,          -2*V,          -2*W,        0,                     0,                       1,             0,     0,     0,     0,    0,    0,    0, 0;...
%             U^2*(V^2 - W^2),                                                                   - 2*U*V^2 + 2*U*W^2,         -2*U^2*V,          2*U^2*W,     4*U*V,    -4*U*W,         0,                     - U^2 + V^2/3 - W^2/3,                         U^2 + V^2/3 - W^2/3,                           V^2/3 - W^2/3,              0,             -V,              W,          -2*U,            -V,             W,        0,                     0,                       0,             1,     0,     0,     0,    0,    0,    0, 0;...
%                     U^2*V*W,                                                                                                           - 2*V*W*U,           -U^2*W,           -U^2*V,     2*U*W,     2*U*V,       U^2,                                   (V*W)/3,                                     (V*W)/3,                                 (V*W)/3,              0,           -W/2,           -V/2,             0,          -W/2,          -V/2,     -2*U,                     0,                       0,             0,     1,     0,     0,    0,    0,    0, 0;...
%                     U*V^2*W,                                                                                                   - V^2*W,         -2*U*V*W,           -U*V^2,     2*V*W,       V^2,     2*U*V,                                -(2*U*W)/3,                                     (U*W)/3,                                 (U*W)/3,           -W/2,              0,           -U/2,          -W/2,             0,           U/2,     -2*V,                     0,                       0,             0,     0,     1,     0,    0,    0,    0, 0;...
%                     U*V*W^2,                                                                                                           - V*W^2,           -U*W^2,         -2*U*V*W,       W^2,     2*V*W,     2*U*W,                                   (U*V)/3,                                  -(2*U*V)/3,                                 (U*V)/3,           -V/2,           -U/2,              0,           V/2,           U/2,             0,     -2*W,                     0,                       0,             0,     0,     0,     1,    0,    0,    0, 0;...
%                  -U*V^2*W^2,                                                                                                V^2*W^2,        2*U*V*W^2,        2*U*V^2*W,  -2*V*W^2,  -2*V^2*W,  -4*U*V*W,                      -(U*(V^2 - 2*W^2))/3,                         (U*(2*V^2 - W^2))/3,                      -(U*(V^2 + W^2))/3,  V^2/2 + W^2/2,            U*V,            U*W, W^2/2 - V^2/2,          -U*V,          -U*W,    4*V*W,                  -U/2,                     U/2,             0,     0,  -2*W,  -2*V,    1,    0,    0, 0;...
%                  -U^2*V*W^2,                                                                                                        -(U*V*W^2*(U - 36028797018963968))/18014398509481984,          U^2*W^2,        2*U^2*V*W,  -2*U*W^2,  -4*U*V*W,  -2*U^2*W,                        -(V*(U^2 + W^2))/3,                         (V*(2*U^2 - W^2))/3,                      -(V*(U^2 + W^2))/3,            U*V,  U^2/2 + W^2/2,            V*W,          -U*V, W^2/2 - U^2/2,           V*W,    4*U*W,                  -V/4,                    -V/4,           V/2,  -2*W,     0,  -2*U,    0,    1,    0, 0;...
%                  -U^2*V^2*W,                                                                                                2*U*V^2*W,        2*U^2*V*W,          U^2*V^2,  -4*U*V*W,  -2*U*V^2,  -2*U^2*V,                       (W*(2*U^2 - V^2))/3,                          -(W*(U^2 + V^2))/3,                      -(W*(U^2 + V^2))/3,            U*W,            V*W,  U^2/2 + V^2/2,           U*W,           V*W, V^2/2 - U^2/2,    4*U*V,                  -W/4,                    -W/4,          -W/2,  -2*V,  -2*U,     0,    0,    0,    1, 0;...
%                 U^2*V^2*W^2,                                                                                               - 2*U*V^2*W^2,     -2*U^2*V*W^2,     -2*U^2*V^2*W, 4*U*V*W^2, 4*U*V^2*W, 4*U^2*V*W, (U^2*V^2)/3 - (2*U^2*W^2)/3 + (V^2*W^2)/3, - (2*U^2*V^2)/3 + (U^2*W^2)/3 + (V^2*W^2)/3, (U^2*V^2)/3 + (U^2*W^2)/3 + (V^2*W^2)/3, -U*(V^2 + W^2), -V*(U^2 + W^2), -W*(U^2 + V^2), U*(V^2 - W^2), V*(U^2 - W^2), W*(U^2 - V^2), -8*U*V*W, U^2/2 + V^2/4 + W^2/4, - U^2/2 + V^2/4 + W^2/4, W^2/2 - V^2/2, 4*V*W, 4*U*W, 4*U*V, -2*U, -2*V, -2*W, 1];
%  

K_eq = simplify(T*feq);
K_force = simplify(T*force');
Id = eye(27,27);
K_pre = sym(zeros(27,1));
syms k4_pre k5_pre k6_pre k7_pre k8_pre real
K_pre(1) = R;
K_pre(5) = k4_pre;
K_pre(6) = k5_pre;
K_pre(7) = k6_pre;
K_pre(8) = k7_pre;
K_pre(9) = k8_pre;
%post-collision central moments
K_star = (Id-L)*K_pre + L*K_eq + (Id-L/2)*K_force


%post collision populations
syms k1 k2 k3 k4 k5 k6 k7 k8 k9 k10 k11 k12 k13 k14 k15 k16 k17 k18 k19...
     k20 k21 k22 k23 k24 k25 k26 real
K_sym = [R k1 k2 k3 k4 k5 k6 k7 k8 k9 k10 k11 k12 0 0 0 0 k17 k18 0 0 0 0 k23 k24 k25 k26];
for i=1:27
    if(K_star(i)~=sym(0))
        K_star(i) = K_sym(i);
    end
end
% f_post_collision_onestep = collect(simplify(T\K_star), K_star);
% 
%% Fei's two-steps approach
raw_moments = collect(simplify(N \ K_star), K_star)
syms r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19...
     r20 r21 r22 r23 r24 r25 r26 real
r = [r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19...
     r20 r21 r22 r23 r24 r25 r26]'; %symbolic raw moments
f_post_collision_twosteps = collect(simplify(M\r),K_star);
f_post_collision_twosteps = collect(f_post_collision_twosteps, 1/8);
f_post_collision_twosteps = collect(f_post_collision_twosteps, -1/8);
f_post_collision_twosteps = collect(f_post_collision_twosteps, 1/4);
f_post_collision_twosteps = collect(f_post_collision_twosteps, -1/4);
f_post_collision_twosteps = collect(f_post_collision_twosteps, 1/2);
f_post_collision_twosteps = collect(f_post_collision_twosteps, -1/2);
f_post_collision_twosteps = collect(f_post_collision_twosteps, 1/6);
f_post_collision_twosteps = collect(f_post_collision_twosteps, 1/16)


%% Charge
syms Q KU KV KW omegaq k1_star k2_star k3_star...
    h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 h17 h18 h19 h20 h21 h22 h23 h24 h25 h26 real
h = [h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 h17 h18 h19 h20 h21 h22 h23 h24 h25 h26];
Lq = diag([1, omegaq, omegaq, omegaq, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
heq = sym(zeros(27,1));
u2 = KU*KU+KV*KV+KW*KW;
for i=1:length(cx)
    first_order = KU*cx(i)+KV*cy(i)+KW*cz(i);
    first_order = first_order/cs2;
    
    second_order = 4.5*(KU*cx(i)+KV*cy(i)+KW*cz(i))^2-1.5*u2;
    
    third_order = (cx(i)^2-cs2)*cy(i)*KU*KU*KV + (cx(i)^2-cs2)*cz(i)*KU*KU*KW +...
                  (cy(i)^2-cs2)*cx(i)*KU*KV*KV + (cz(i)^2-cs2)*cx(i)*KU*KW*KW +...
                  (cz(i)^2-cs2)*cy(i)*KV*KW*KW + (cy(i)^2-cs2)*cz(i)*KV*KV*KW +...
              2*( cx(i)*cy(i)*cz(i)*KU*KV*KW);  
    third_order = third_order/(2*cs6);

    fourth_order = (cx(i)^2-cs2)*(cy(i)^2-cs2)*KU*KU*KV*KV +...
                   (cx(i)^2-cs2)*(cz(i)^2-cs2)*KU*KU*KW*KW +...
                   (cy(i)^2-cs2)*(cz(i)^2-cs2)*KV*KV*KW*KW +...
                2*( cx(i)*cy(i)*(cz(i)^2-cs2)*KU*KV*KW*KW +...
                    cx(i)*(cy(i)^2-cs2)*cz(i)*KU*KV*KV*KW +...
                   (cx(i)^2-cs2)*cy(i)*cz(i)*KU*KU*KV*KW );
    fourth_order = fourth_order/(4*cs8);
    
    fifth_order = (cx(i)^2-cs2)*cy(i)*(cz(i)^2-cs2)*KU*KU*KV*KW*KW +...
                  (cx(i)^2-cs2)*(cy(i)^2-cs2)*cz(i)*KU*KU*KV*KV*KW +...
                  cx(i)*(cy(i)^2-cs2)*(cz(i)^2-cs2)*KU*KV*KV*KW*KW;
    fifth_order = fifth_order/(4*cs10);

    sixth_order = (cx(i)^2-cs2)*(cy(i)^2-cs2)*(cz(i)^2-cs2)*KU*KU*KV*KV*KW*KW;
    sixth_order = sixth_order/(8*cs12);
    heq(i) = Q*w(i)*(1+first_order + second_order + third_order +...
                         fourth_order + fifth_order + sixth_order);

    % Set the basis
    CX = cx(i)-KU;
    CY = cy(i)-KV;
    CZ = cz(i)-KW;
    T(1,i) = 1;
    T(2,i) = CX;
    T(3,i) = CY;
    T(4,i) = CZ;
    T(5,i) = CX*CY;
    T(6,i) = CX*CZ;
    T(7,i) = CY*CZ;
    T(8,i) = CX*CX-CY*CY;
    T(9,i) = CX*CX-CZ*CZ;
    T(10,i) = CX*CX+CY*CY+CZ*CZ;
    T(11,i) = CX*CY*CY+CX*CZ*CZ;
    T(12,i) = CX*CX*CY+CY*CZ*CZ;
    T(13,i) = CX*CX*CZ+CY*CY*CZ;
    T(14,i) = CX*CY*CY-CX*CZ*CZ;
    T(15,i) = CX*CX*CY-CY*CZ*CZ;
    T(16,i) = CX*CX*CZ-CY*CY*CZ;
    T(17,i) = CX*CY*CZ;
    T(18,i) = CX*CX*CY*CY+CX*CX*CZ*CZ+CY*CY*CZ*CZ;
    T(19,i) = CX*CX*CY*CY+CX*CX*CZ*CZ-CY*CY*CZ*CZ;
    T(20,i) = CX*CX*CY*CY-CX*CX*CZ*CZ;
    T(21,i) = CX*CX*CY*CZ;
    T(22,i) = CX*CY*CY*CZ;
    T(23,i) = CX*CY*CZ*CZ;
    T(24,i) = CX*CY*CY*CZ*CZ;
    T(25,i) = CX*CX*CY*CZ*CZ;
    T(26,i) = CX*CX*CY*CY*CZ;
    T(27,i) = CX*CX*CY*CY*CZ*CZ;
    
    CX = cx(i);
    CY = cy(i);
    CZ = cz(i);
    M(1,i) = 1;
    M(2,i) = CX;
    M(3,i) = CY;
    M(4,i) = CZ;
    M(5,i) = CX*CY;
    M(6,i) = CX*CZ;
    M(7,i) = CY*CZ;
    M(8,i) = CX*CX-CY*CY;
    M(9,i) = CX*CX-CZ*CZ;
    M(10,i) = CX*CX+CY*CY+CZ*CZ;
    M(11,i) = CX*CY*CY+CX*CZ*CZ;
    M(12,i) = CX*CX*CY+CY*CZ*CZ;
    M(13,i) = CX*CX*CZ+CY*CY*CZ;
    M(14,i) = CX*CY*CY-CX*CZ*CZ;
    M(15,i) = CX*CX*CY-CY*CZ*CZ;
    M(16,i) = CX*CX*CZ-CY*CY*CZ;
    M(17,i) = CX*CY*CZ;
    M(18,i) = CX*CX*CY*CY+CX*CX*CZ*CZ+CY*CY*CZ*CZ;
    M(19,i) = CX*CX*CY*CY+CX*CX*CZ*CZ-CY*CY*CZ*CZ;
    M(20,i) = CX*CX*CY*CY-CX*CX*CZ*CZ;
    M(21,i) = CX*CX*CY*CZ;
    M(22,i) = CX*CY*CY*CZ;
    M(23,i) = CX*CY*CZ*CZ;
    M(24,i) = CX*CY*CY*CZ*CZ;
    M(25,i) = CX*CX*CY*CZ*CZ;
    M(26,i) = CX*CX*CY*CY*CZ;
    M(27,i) = CX*CX*CY*CY*CZ*CZ;
end
% Compute central moments
T = simplify(T);
N = simplify(T*M^(-1)); %shift matrix
Kq_eq = simplify(T*heq);
Kq_pre = sym(zeros(27,1));
syms kq1_pre kq2_pre kq3_pre real
Kq_pre(1) = Q;
Kq_pre(2) = kq1_pre;
Kq_pre(3) = kq2_pre;
Kq_pre(4) = kq3_pre;
%post-collision central moments
Kq_star = (Id-Lq)*Kq_pre + Lq*Kq_eq

%post collision populations
syms kq1 kq2 kq3 kq4 kq5 kq6 kq7 kq8 kq9 kq10 kq11 kq12 kq13 kq14 kq15 kq16 kq17 kq18 kq19...
     kq20 kq21 kq22 kq23 kq24 kq25 kq26 real
Kq_sym = [Q kq1 kq2 kq3 kq4 kq5 kq6 kq7 kq8 kq9 kq10 kq11 kq12 kq13 kq14 kq15 kq16 kq17 kq18 kq19...
     kq20 kq21 kq22 kq23 kq24 kq25 kq26];
for i=1:27
    if(Kq_star(i)~=sym(0))
        Kq_star(i) = Kq_sym(i);
    end
end
%% Fei's two-steps approach
rq = collect(simplify(N^(-1) * Kq_star), Kq_star)
syms Q rq1 rq2 rq3 rq4 rq5 rq6 rq7 rq8 rq9 rq10 rq11 rq12 rq13 rq14 rq15 rq16 rq17 rq18 rq19...
     rq20 rq21 rq22 rq23 rq24 rq25 rq26 real
rq = [Q rq1 rq2 rq3 rq4 rq5 rq6 rq7 rq8 rq9 rq10 rq11 rq12 rq13 rq14 rq15 rq16 rq17 rq18 rq19...
     rq20 rq21 rq22 rq23 rq24 rq25 rq26]'; %symbolic raw moments
h_post_collision_twosteps = collect(simplify(M\rq),Kq_star)