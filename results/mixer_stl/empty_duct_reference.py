# Thin-interface (Leveque-type) flux-weighted mixing index in a fully developed square duct,
# two streams split at y = W/2:  c = (1 + erf((y - W/2) / (2 sqrt(D x / u_i(z))))) / 2,
# u_i(z) = u(W/2, z).  int c(1-c) dy = sqrt(2/pi) sqrt(D x / u_i)  (exact for the erf profile),
# so  Phi = 1/4 - sqrt(2/pi) sqrt(D x) int sqrt(u_i) dz / Q,  M = 1 - sqrt(4 Phi).
# Valid while the interface is thin against W; axial diffusion neglected (Pe = 2000).
import math
def K():
    s=sum(math.tanh(i*math.pi/2)/i**5 for i in range(1,4000,2)); return 12/(1-192/math.pi**5*s)
def f(y,z,KK):  # u/Ubar, y,z in [0,1]
    a=0.5; yc=y-a; zc=abs(z-a); S=0
    for i in range(1,1200,2):
        k=i*math.pi/(2*a); r=math.exp(k*(zc-a))*(1+math.exp(-2*k*zc))/(1+math.exp(-2*k*a))
        S+=(1 if ((i-1)//2)%2==0 else -1)*(1-r)*math.cos(k*yc)/i**3
    return 4*KK/math.pi**3*S
KK=K(); W=200e-6; U=0.01; D=1e-9
nq=400; I=sum(math.sqrt(max(f(0.5,(j+0.5)/nq,KK),0)) for j in range(nq))/nq   # <sqrt(f)> over z
# check: the quadrature of 1-erf^2
h=1e-3; q=sum((1-math.erf(-20+(k+0.5)*h)**2)*h for k in range(int(40/h)))
print("int (1-erf^2) ds = %.6f  (2 sqrt(2/pi) = %.6f);  u_max/U = %.4f;  <sqrt(u_i/U)> = %.5f"%(q,2*math.sqrt(2/math.pi),f(.5,.5,KK),I))
for x in (100e-6,200e-6,400e-6,600e-6,800e-6,880e-6):
    phi=0.25-math.sqrt(2/math.pi)*math.sqrt(D*x)*math.sqrt(U)*I*W/(U*W*W)
    print("x = %4.0f um   interface 2 sqrt(Dx/U) = %5.2f um   M_flux = %.5f"%(x*1e6,2*math.sqrt(D*x/U)*1e6,1-math.sqrt(4*phi)))
