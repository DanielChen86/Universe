import numpy as np
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


class Universe(Parameter, InitialValue):
    def __init__(self, parameter, ivalue) -> None:
        self.t = 0

        Parameter.__init__(self, parameter)
        InitialValue.__init__(self, ivalue)

        self.alpha = self.parameter['g']**2 / (4 * np.pi)
        self.g_tilde = self.parameter['Nc']**2 - 1

        self.recorder = {'t': [], 'a': [], 'a_dot': [], 'H': []}
        self.recorder['t'] = []
        self.recorder['a'] = []
        self.recorder['a_dot'] = []
        self.recorder['H'] = []
        self.recorder['phi_field'] = []
        self.recorder['phi_field_dot'] = []
        self.recorder['phi_field_dot_dot'] = []
        self.recorder['phi_temperature'] = []
        self.recorder['phi_density'] = []
        self.recorder['dark_radiation_density'] = []
        self.recorder['Upsilon'] = []

        self.phi_temperature = self.ivalue['phi_temperature']
        self.dark_radiation_temperature = self.ivalue['phi_temperature']

        self.phi_field = self.ivalue['phi_field']
        self.phi_field_dot = self.ivalue['phi_field_dot']

        self.a = self.ivalue['a']
        self.a_dot = self.get_a_dot()

    def get_Gamma_sph(self):  # maybe need modification
        return self.parameter['Nc']**5 * self.alpha**5 * self.phi_temperature**4

    def get_Upsilon(self):
        return self.get_Gamma_sph() / (2 * self.phi_temperature * self.parameter['f']**2)

    def get_dark_radiation_density(self):
        return (2 * self.parameter['Nc']**2 - 1) * (np.pi**2 / 30) * self.dark_radiation_temperature**4

    def get_phi_potential(self):
        return - self.parameter['C'] * self.phi_field

    def get_phi_density(self):
        density = (1/2) * self.phi_field_dot**2 + self.get_phi_potential()
        if density < 0:
            return 0
        return density

    def EOM_dark_radiation_temperature_dot(self):
        factor = (2 * self.parameter['Nc']**2 - 1) * (np.pi**2 / 30)
        return - self.H * self.dark_radiation_temperature + self.get_Upsilon() * self.phi_field_dot**2 / (4 * self.dark_radiation_temperature**3 * factor)

    def get_a_dot(self):
        total_density = self.get_phi_density() + self.get_dark_radiation_density()# + self.get_G_density()
        pi_rho = (8 * np.pi * self.parameter['G'] / 3) * total_density
        pi_rho = max(0, pi_rho)
        return np.sqrt(pi_rho) * self.a

    def run(self, dt=-1e-10, iteration: int = 10):
        for _ in range(iteration):
            self.t += dt
            self.H = self.a_dot / self.a

            self.dark_radiation_temperature += self.EOM_dark_radiation_temperature_dot() * dt
            self.phi_temperature = self.dark_radiation_temperature

            self.phi_field_dot_dot = -3 * self.H * self.phi_field_dot - \
                self.get_Upsilon() * self.phi_field_dot + self.parameter['C']

            self.dark_radiation_density = self.get_dark_radiation_density()

            self.phi_field_dot += self.phi_field_dot_dot * dt
            self.phi_field += self.phi_field_dot * dt

            self.a_dot = self.get_a_dot()
            self.a += self.a_dot * dt

            self.record()
        for k, v in self.recorder.items():
            self.recorder[k] = np.array(v)

    def record(self):
        self.recorder['t'].append(self.t)
        self.recorder['a'].append(self.a)
        self.recorder['a_dot'].append(self.a_dot)
        self.recorder['H'].append(self.H)

        self.recorder['phi_field'].append(self.phi_field)
        self.recorder['phi_field_dot'].append(self.phi_field_dot)
        self.recorder['phi_field_dot_dot'].append(self.phi_field_dot_dot)
        self.recorder['phi_density'].append(self.get_phi_density())
        self.recorder['dark_radiation_density'].append(
            self.get_dark_radiation_density())
        self.recorder['phi_temperature'].append(self.phi_temperature)
        self.recorder['Upsilon'].append(self.get_Upsilon())


if __name__ == '__main__':
    universe = Universe('Parameter.yaml', 'InitialValueNew.yaml')
    universe.run(dt=1e-4, iteration=10000)
    fig, axs = plt.subplots(4, 2, figsize=(15, 16))

    axs[0, 0].scatter(universe.recorder['t'],
                      universe.recorder['a'], s=3, label='a')
    axs[0, 0].legend()

    axs[0, 1].scatter(universe.recorder['t'],
                      universe.recorder['a_dot'], s=3, label='a_dot')
    axs[0, 1].legend()

    axs[1, 0].scatter(universe.recorder['t'],
                      universe.recorder['phi_temperature'], s=3, label='phi_temperature')
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
    axs[2, 0].legend()

    axs[2, 1].scatter(universe.recorder['t'],
                      universe.recorder['phi_field'], s=3, label='phi')
    axs[2, 1].scatter(universe.recorder['t'],
                      universe.recorder['phi_field_dot'], s=3, label='phi_dot')
    axs[2, 1].scatter(universe.recorder['t'], universe.recorder['phi_field_dot_dot'] /
                      universe.recorder['H'], s=3, label='phi_dot_dot')
    axs[2, 1].legend()


    plt.tight_layout()
    plt.show()
