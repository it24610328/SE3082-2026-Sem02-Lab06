#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 200
#define BOXL 10.0
#define DT 0.001
#define STEPS 50

double x[N], y[N], z[N];
double vx[N], vy[N], vz[N];
double fx[N], fy[N], fz[N];

double epot_total, virial_total;

void initialize() {
    int n = (int)cbrt((double)N) + 1;
    double spacing = BOXL / n;
    int idx = 0;
    for (int i = 0; i < n && idx < N; i++) {
        for (int j = 0; j < n && idx < N; j++) {
            for (int k = 0; k < n && idx < N; k++) {
                x[idx] = i * spacing;
                y[idx] = j * spacing;
                z[idx] = k * spacing;
                vx[idx] = ((rand() % 200) - 100) / 1000.0;
                vy[idx] = ((rand() % 200) - 100) / 1000.0;
                vz[idx] = ((rand() % 200) - 100) / 1000.0;
                idx++;
            }
        }
    }
}

void domove() {
    for (int i = 0; i < N; i++) {
        x[i] += vx[i] * DT + 0.5 * fx[i] * DT * DT;
        y[i] += vy[i] * DT + 0.5 * fy[i] * DT * DT;
        z[i] += vz[i] * DT + 0.5 * fz[i] * DT * DT;

        if (x[i] < 0) x[i] += BOXL;
        if (x[i] >= BOXL) x[i] -= BOXL;
        if (y[i] < 0) y[i] += BOXL;
        if (y[i] >= BOXL) y[i] -= BOXL;
        if (z[i] < 0) z[i] += BOXL;
        if (z[i] >= BOXL) z[i] -= BOXL;

        vx[i] += 0.5 * fx[i] * DT;
        vy[i] += 0.5 * fy[i] * DT;
        vz[i] += 0.5 * fz[i] * DT;
    }
}

void forces() {
    double epot = 0.0, virial = 0.0;

    for (int i = 0; i < N; i++) {
        fx[i] = 0.0; fy[i] = 0.0; fz[i] = 0.0;
    }

    #pragma omp parallel for reduction(+:epot, virial) schedule(static, 4)
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            double dx = x[i] - x[j];
            double dy = y[i] - y[j];
            double dz = z[i] - z[j];

            dx -= BOXL * round(dx / BOXL);
            dy -= BOXL * round(dy / BOXL);
            dz -= BOXL * round(dz / BOXL);

            double r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < 9.0 && r2 > 0.0001) {
                double r2inv = 1.0 / r2;
                double r6inv = r2inv * r2inv * r2inv;
                double ffac = 48.0 * r2inv * r6inv * (r6inv - 0.5);

                epot += 4.0 * r6inv * (r6inv - 1.0);
                virial += ffac * r2;

                #pragma omp critical
                {
                    fx[i] += ffac * dx;
                    fy[i] += ffac * dy;
                    fz[i] += ffac * dz;
                    fx[j] -= ffac * dx;
                    fy[j] -= ffac * dy;
                    fz[j] -= ffac * dz;
                }
            }
        }
    }

    epot_total = epot;
    virial_total = virial;
}

double mkekin() {
    double ekin = 0.0;
    for (int i = 0; i < N; i++) {
        ekin += 0.5 * (vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]);
    }
    return ekin;
}

double velavg() {
    double vsum = 0.0;
    for (int i = 0; i < N; i++) {
        vsum += sqrt(vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]);
    }
    return vsum / N;
}

void prnout(int step, double epot, double ekin, double virial, double vavg) {
    printf("Step %3d | PE = %10.4f | KE = %10.4f | Virial = %10.4f | AvgVel = %8.4f\n",
           step, epot, ekin, virial, vavg);
}

int main() {
    double tstart, tstop, tcalc;
    srand(42);
    initialize();

    tstart = omp_get_wtime();

    for (int step = 0; step < STEPS; step++) {
        domove();
        forces();
        double ekin = mkekin();
        double vavg = velavg();
        if (step % 10 == 0) {
            prnout(step, epot_total, ekin, virial_total, vavg);
        }
    }

    tstop = omp_get_wtime();
    tcalc = tstop - tstart;

    printf("Threads used: %d\n", omp_get_max_threads());
    printf("Total time = %f seconds\n", tcalc);

    return 0;
}
