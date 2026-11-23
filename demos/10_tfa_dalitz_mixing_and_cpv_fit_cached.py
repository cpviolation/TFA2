import amplitf.interface as atfi
import amplitf.kinematics as atfk
import amplitf.dynamics as atfd
import amplitf.likelihood as atfl
import amplitf.mixing as atfm
from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace
from amplitf.phasespace.decaytime_phasespace import DecayTimePhaseSpace
from amplitf.phasespace.combined_phasespace import CombinedPhaseSpace

import tfa.toymc as tft
import tfa.plotting as tfp
import tfa.optimisation as tfo

ntoys = 100000  # Number of points to generate
nnorm = 10*ntoys  # Number of normalisation points

mk = atfi.const(0.498)
mpi = atfi.const(0.139)
md = atfi.const(1.8646)

rd = atfi.const(5.0)
rr = atfi.const(1.5)

mkst = atfi.const(0.892)
wkst = atfi.const(0.050)
mrho = atfi.const(0.770)
wrho = atfi.const(0.150)

xmix = atfi.const(0.04) # x10 to show effects with small sample size
ymix = atfi.const(0.064)  # x10 to show effects with small sample size
qop = atfi.const(1.0)
qop_phase = atfi.const(0.0)  # radians

phsp = DalitzPhaseSpace(mpi, mk, mpi, md)
tdz = atfi.const(1.0)
tphsp = DecayTimePhaseSpace(tdz)
c_phsp = CombinedPhaseSpace(phsp, tphsp)


def model_dz_amp(x):

    m2ab = phsp.m2ab(x)
    m2bc = phsp.m2bc(x)
    m2ac = phsp.m2ac(x)

    hel_ab = atfd.helicity_amplitude(phsp.cos_helicity_ab(x), 1)
    hel_bc = atfd.helicity_amplitude(phsp.cos_helicity_bc(x), 1)
    hel_ac = atfd.helicity_amplitude(phsp.cos_helicity_ac(x), 1)

    bw1 = atfd.breit_wigner_lineshape(m2ab, mkst, wkst, mpi, mk, mpi, md, rd, rr, 1, 1)
    bw2 = atfd.breit_wigner_lineshape(m2bc, mkst, wkst, mpi, mk, mpi, md, rd, rr, 1, 1)
    bw3 = atfd.breit_wigner_lineshape(m2ac, mrho, wrho, mpi, mpi, mk, md, rd, rr, 1, 1)

    def _model(a1r, a1i, a2r, a2i, a3r, a3i, switches=4 * [1]):

        a1 = atfi.complex(a1r, a1i)
        a2 = atfi.complex(a2r, a2i)
        a3 = atfi.complex(a3r, a3i)

        ampl = atfi.cast_complex(atfi.ones(m2ab)) * atfi.complex(
            atfi.const(0.0), atfi.const(0.0)
        )

        if switches[0]:
            ampl += a1 * bw1 * hel_ab
        if switches[1]:
            ampl += a2 * bw2 * hel_bc
        if switches[2]:
            ampl += a3 * bw3 * hel_ac
        if switches[3]:
            ampl += atfi.cast_complex(atfi.ones(m2ab)) * atfi.complex(
                atfi.const(5.0), atfi.const(0.0)
            )

        return ampl

    return _model


def Af(x):
    return model_dz_amp(x)(
        a1r=atfi.const(1.0),
        a1i=atfi.const(0.0),
        a2r=atfi.const(0.5),
        a2i=atfi.const(0.0),
        a3r=atfi.const(2.0),
        a3i=atfi.const(0.0),
        switches=4 * [1],
    )

def Afbar(x):
    return Af(x[:,::-1])


def mixing_model(x, flavor=True):
    """Time-dependent density for an initial D0 (flavor=True) or D0bar (flavor=False)."""
    # Calculate the amplitudes - cached
    ampl_dz = Af(c_phsp.data1(x))
    ampl_dzb = Afbar(c_phsp.data1(x))
    # for an initial D0bar the roles of A_f and Abar_f are exchanged (and q/p -> p/q)
    ampl_dir, ampl_cnj = (ampl_dz, ampl_dzb) if flavor else (ampl_dzb, ampl_dz)
    # Calculate the model from the mixing parameters and time evolution operators
    def _model(x_mix_par, y_mix_par, qop_alpha_par, qop_phase_par):
        t = c_phsp.phsp2.t(c_phsp.data2(x))
        # DecayTimePhaseSpace draws t from exp(-t/tau): use the time evolution
        # relative to that exponential, both for the toys and for the integrals
        tep = atfm.psip(t, y_mix_par, tdz, relative_to_decay=True)
        tem = atfm.psim(t, y_mix_par, tdz, relative_to_decay=True)
        tei = atfm.psii(t, x_mix_par, tdz, relative_to_decay=True)
        # calculate the amplitude density
        qop_par = atfi.tan(qop_alpha_par)
        qoverp = atfi.complex(qop_par * atfi.cos(qop_phase_par), qop_par * atfi.sin(qop_phase_par))
        dens = atfm.mixing_density(ampl_dir, ampl_cnj, qoverp, tep, tem, tei, flavor=flavor)
        return dens
    return _model


def toymc_model(flavor):
    def _pdf(x):
        return mixing_model(x, flavor)(
            x_mix_par=xmix,
            y_mix_par=ymix,
            qop_alpha_par=atfi.atan(qop),
            qop_phase_par=qop_phase
        )
    return _pdf


# TF graph for unbinned negalite log likelihood (the quantity to be minimised).
# D0 and D0bar decays are needed together: with D0 decays only, |q/p| and (x, y)
# are nearly degenerate. Each flavour is normalised separately (independent yields).
def nll(data_dz, data_dzb, norm):
    models = [(mixing_model(data_dz, True), mixing_model(norm, True)),
              (mixing_model(data_dzb, False), mixing_model(norm, False))]

    @atfi.function
    def _nll(pars):
        return sum(atfl.unbinned_nll(data_model(**pars), atfl.integral(norm_model(**pars)))
                   for data_model, norm_model in models)

    return _nll


toy_sample_dz = tft.run_toymc(
    toymc_model(True), c_phsp, ntoys, maximum=1.0e-20, chunk=1000000, components=False
)
toy_sample_dzb = tft.run_toymc(
    toymc_model(False), c_phsp, ntoys, maximum=1.0e-20, chunk=1000000, components=False
)

print(toy_sample_dz)
print(toy_sample_dzb)

norm_sample = c_phsp.uniform_sample(nnorm)

print(norm_sample)

pars = [
    tfo.FitParameter("x_mix_par", 0.04, -0.2, 0.2),
    tfo.FitParameter("y_mix_par", 0.06, -0.2, 0.2),
    tfo.FitParameter("qop_alpha_par", atfi.pi()/4, 0., atfi.pi()/2),
    tfo.FitParameter("qop_phase_par", 0.0, -3.15, 3.15),
]
# fixed_pars = ['qop_par','qop_phase_par']  # Fix qop and qop_phase for this example
# for p in pars:
#     if p.name in fixed_pars:
#         p.fix()

# Run MINUIT minimisation of the neg. log likelihood
result = tfo.run_minuit(nll(toy_sample_dz, toy_sample_dzb, norm_sample), pars)
print(result)

cov = result['covariance']

print(f"{result['time']/result['func_calls']} sec per function call")

fitted_pars = {p: atfi.const(v[0]) for p, v in result["params"].items()}


def fitted_model(flavor):
    def _pdf(x):
        return mixing_model(x, flavor)(**fitted_pars)
    return _pdf


# ff = tfo.calculate_fit_fractions(fitted_model, norm_sample)
# print(ff)

# Plot D0 and D0bar decays together
toy_sample = atfi.concat([toy_sample_dz, toy_sample_dzb], axis=0)
fitted_sample = atfi.concat([
    tft.run_toymc(fitted_model(flavor), c_phsp, nnorm // 2, maximum=1.0e-20, chunk=1000000, components=False)
    for flavor in (True, False)
], axis=0)

# Plot results
import matplotlib.pyplot as plt

tfp.set_lhcb_style(size=12, usetex=False)  # Adjust plotting style for LHCb papers
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))  # Single subplot on the figure

# Plot 1D histogram from the toy MC sample
tfp.plot_distr2d(
    toy_sample[:, 0],
    toy_sample[:, 1],
    bins=(50, 50),
    ranges=((0.3, 3.1), (0.3, 3.1)),
    fig=fig,
    ax=ax[0, 0],
    labels=(r"$m^2(K_S^0\pi^+)$", r"$m^2(K_S^0\pi^-)$"),
    units=("MeV$^2$", "MeV$^2$"),
    log=True,
)

tfp.plot_distr1d_comparison(
    toy_sample[:, 0],
    fitted_sample[:, 0],
    #cweights=[fitted_sample[:, 3 + i] for i in range(4)],
    bins=50,
    range=(0.3, 3.1),
    ax=ax[0, 1],
    label=r"$m^2(K_S^0\pi^+)$",
    units="MeV$^2$",
    legend=False
)

tfp.plot_distr1d_comparison(
    toy_sample[:, 1],
    fitted_sample[:, 1],
    #cweights=[fitted_sample[:, 3 + i] for i in range(4)],
    bins=50,
    range=(0.3, 3.1),
    ax=ax[1, 0],
    label=r"$m^2(K_S^0\pi^-)$",
    units="MeV$^2$",
    legend=False
)

tfp.plot_distr1d_comparison(
    toy_sample[:, 2],
    fitted_sample[:, 2],
    #cweights=[fitted_sample[:, 3 + i] for i in range(4)],
    bins=50,
    range=(0., 5),
    ax=ax[0, 2],
    label=r"t",
    units=r"$\tau_{D^0}$",
    log=True,
    legend=False
)

tfp.plot_distr1d_comparison(
    c_phsp.phsp1.m2ac(c_phsp.data1(toy_sample)),
    c_phsp.phsp1.m2ac(c_phsp.data1(fitted_sample)),
    #cweights=[fitted_sample[:, 3 + i] for i in range(4)],
    bins=50,
    range=(0.05, 1.9),
    ax=ax[1, 1],
    label=r"$m^2(\pi^+\pi^-)$",
    units="MeV$^2$",
    legend_ax=ax[1,2]
)

# Show the plot
plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
plt.show()
