import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from scipy.special import zeta


class Parameter:
    def __init__(self, parameter) -> None:
        with open(parameter, 'r') as f:
            self.parameter = yaml.safe_load(f)


class InitialValue:
    def __init__(self, ivalue) -> None:
        with open(ivalue, 'r') as f:
            self.ivalue = yaml.safe_load(f)

class Constant:
    def __init__(self) -> None:
        self.h = 4.13566769 * 1e-15
        self.k = 8.617333262 * 1e-5
        self.c = 3 * 1e8
        self.eV = 1.6 * 1e-19

class Universe(Parameter, InitialValue):
    def __init__(self, parameter, ivalue) -> None:

        Parameter.__init__(self, parameter)
        InitialValue.__init__(self, ivalue)

        self.constant = Constant()

        self.Omega_m = self.ivalue['Omega_m']
        self.Omega_Lambda = self.ivalue['Omega_Lambda']
        self.H0 = self.ivalue['H0']

        self.alpha = self.parameter['g']**2 / (4 * np.pi)
        self.g_tilde = self.parameter['Nc']**2 - 1
        self.TR = self.parameter['Nc']  # adjoint representation
        self.dR = self.parameter['Nc']  # need modification
        self.kappa = 1  # need modification
        self.mfermion = self.parameter['mfermion']

        self.recorder = {}
        self.recorder['t'] = []
        self.recorder['a'] = []
        self.recorder['H'] = []
        self.recorder['phi_field'] = []
        self.recorder['phi_field_dot'] = []
        self.recorder['phi_field_dot_dot'] = []
        self.recorder['temperature'] = []
        self.recorder['phi_density'] = []
        self.recorder['dark_radiation_density'] = []
        self.recorder['G_field'] = []
        self.recorder['Upsilon'] = []
        self.recorder['nR'] = []
        self.recorder['nR_dot'] = []
        self.recorder['N_field'] = []
        self.recorder['N_field_dot'] = []
        self.recorder['N_density'] = []
        self.recorder['total_density'] = []

        
        
        # self.temperature = 2.726

        self.t = self.ivalue['t0']
        self.temperature = self.ivalue['temperature']
        self.phi_field = self.ivalue['phi_field']
        self.phi_field_dot = self.ivalue['phi_field_dot']
        self.nR = self.ivalue['nR']
        self.N_field = self.ivalue['N_field']


    def a(self):
        return (self.Omega_m / self.Omega_Lambda)**(1/3) * (np.sinh((3/2) * np.sqrt(self.Omega_Lambda) * self.H0 * self.t))**(2/3)
    
    def H(self):
        return np.sqrt(self.Omega_Lambda) * self.H0 * np.cosh((3/2) * np.sqrt(self.Omega_Lambda) * self.H0 * self.t) / np.sinh((3/2) * np.sqrt(self.Omega_Lambda) * self.H0 * self.t)

    def get_Gamma_sph(self):  # maybe need modification
        return self.parameter['Nc']**5 * self.alpha**5 * self.constant.k**4 * self.temperature**4

    def get_Upsilon(self):
        return self.get_Gamma_sph() / (2 * self.constant.k * self.temperature * self.parameter['f']**2)

    def get_dark_radiation_density(self):
        return (2 * self.parameter['Nc']**2 - 1) * (5.67 * self.constant.eV) * self.temperature**4

    def get_phi_potential(self):
        return self.parameter['V0'] - self.parameter['C'] * self.phi_field

    def get_phi_density(self):
        density = (1/2) * self.phi_field_dot**2 + self.get_phi_potential()
        return density
    
    def EOM_dark_radiation_temperature_dot(self):
        factor = (2 * self.parameter['Nc']**2 - 1) * (np.pi**2 / 30)
        return - self.H() * self.constant.k * self.temperature + self.get_Upsilon() * self.phi_field_dot**2 / (4 * self.constant.k**3 * self.temperature**3 * factor)
    
    def get_G_field(self):
        EB = np.pi**2 * self.get_Gamma_sph() * self.phi_field_dot / self.constant.k / self.temperature / self.parameter['f']
        return np.sqrt(EB)
    
    def run(self, dt=-1e-10, iteration: int = 10):
        for _ in range(iteration):
            self.t += dt

            self.temperature += self.EOM_dark_radiation_temperature_dot() * dt

            self.phi_field_dot_dot = - 3 * self.H() * self.phi_field_dot \
                                     - self.get_Gamma_sph() / (2 * self.constant.k * self.temperature * self.parameter['f']**2) * self.phi_field_dot \
                                     + (24 * self.get_Gamma_sph() * self.TR) / (2 * self.parameter['f'] * self.dR) / (self.constant.k**3 * self.temperature**3) * self.nR \
                                     + self.parameter['C']
            self.nR_dot = self.TR * self.get_Gamma_sph() / self.constant.k / self.temperature * (self.phi_field_dot / self.parameter['f'] - (24 * self.TR * self.nR) / (self.dR * self.temperature**2)) \
                          - self.kappa * (self.nR * self.parameter['Nc'] * self.alpha * self.mfermion**2) / self.temperature /self.constant.k
            
            self.nR += self.nR_dot * dt

            self.G_field = self.get_G_field()

            self.N_field_dot = - 3 * self.H() * self.N_field + (1/self.parameter['fN']) * self.g_tilde * self.G_field * np.sqrt(self.nR)
            
            self.N_field += self.N_field_dot * dt

            
            self.N_density = self.N_field**2
            

            self.dark_radiation_density = self.get_dark_radiation_density()

            self.phi_field_dot += self.phi_field_dot_dot * dt
            self.phi_field += self.phi_field_dot * dt

            self.total_density = self.get_phi_density() + self.get_dark_radiation_density() + self.nR + self.N_density


            

            self.record()



            if self.a() > 1:
                print('a > 1')
                break

        for k, v in self.recorder.items():
            self.recorder[k] = np.array(v)

    def record(self):
        self.recorder['t'].append(self.t)
        self.recorder['a'].append(self.a())
        self.recorder['H'].append(self.H())

        self.recorder['phi_field'].append(self.phi_field)
        self.recorder['phi_field_dot'].append(self.phi_field_dot)
        self.recorder['phi_field_dot_dot'].append(self.phi_field_dot_dot)
        self.recorder['phi_density'].append(self.get_phi_density())
        self.recorder['dark_radiation_density'].append(
            self.get_dark_radiation_density())
        self.recorder['temperature'].append(self.temperature)
        self.recorder['Upsilon'].append(self.get_Upsilon())
        self.recorder['nR'].append(self.nR)
        self.recorder['nR_dot'].append(self.nR_dot)
        self.recorder['N_field'].append(self.N_field)
        self.recorder['N_field_dot'].append(self.N_field_dot)
        self.recorder['N_density'].append(self.N_density)
        self.recorder['total_density'].append(self.total_density)
        self.recorder['G_field'].append(self.G_field)


if __name__ == '__main__':

    universe = Universe('Parameter.yaml', 'InitialValueNew.yaml')
    universe.run(dt=1e-4, iteration=10000)
    fig, axs = plt.subplots(4, 2, figsize=(15, 16))

    axs[0, 0].scatter(universe.recorder['t'],
                        universe.recorder['a'], s=3, label='a')
    axs[0, 0].legend()

    # axs[0, 1].scatter(universe.recorder['t'],
    #                     universe.recorder['a_dot'], s=3, label='a_dot')
    # axs[0, 1].legend()

    axs[1, 0].scatter(universe.recorder['t'],
                        universe.recorder['temperature'], s=3, label='temperature')
    axs[1, 0].legend()

    axs[1, 1].scatter(universe.recorder['t'],
                        universe.recorder['Upsilon'], s=3, label='Upsilon')
    axs[1, 1].scatter(universe.recorder['t'],
                        universe.recorder['H'], s=3, label='H')
    axs[1, 1].legend()

    axs[2, 0].scatter(universe.recorder['t'],
                        universe.recorder['phi_density'], s=3, label='phi_density')
    axs[2, 0].scatter(universe.recorder['t'], universe.recorder['dark_radiation_density'],
                    s=3, label='dark_radiation_density')
    # axs[2, 0].scatter(universe.recorder['t'], universe.recorder['total_density'],
    #                 s=3, label='total_density')
    axs[2, 0].legend()

    axs[2, 1].scatter(universe.recorder['t'],
                        universe.recorder['phi_field'], s=3, label='phi')
    axs[2, 1].scatter(universe.recorder['t'],
                        universe.recorder['phi_field_dot'], s=3, label='phi_dot')
    axs[2, 1].scatter(universe.recorder['t'], universe.recorder['phi_field_dot_dot'] /
                        universe.recorder['H'], s=3, label='phi_dot_dot')
    axs[2, 1].legend()

    axs[3, 0].scatter(universe.recorder['t'],
                        universe.recorder['nR'], s=3, label='nR')
    axs[3, 0].legend()

    axs[3, 1].scatter(universe.recorder['t'],
                        universe.recorder['N_field'], s=3, label='N_field')
    axs[3, 1].legend()
    print(universe.recorder['N_field'])

    plt.show()


