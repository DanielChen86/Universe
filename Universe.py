import numpy as np
import yaml


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
        Parameter.__init__(self, parameter)
        InitialValue.__init__(self, ivalue)
        self.a = self.ivalue['a']
        self.phi = Phi(parameter, ivalue)

        self.recorder = {'t': [], 'a': [], 'a_dot': [], 'H': []}
        self.t = self.ivalue['t']

        self.phi_start = False

    def run(self, dt=-1e-10, iteration: int = 10):
        for _ in range(iteration):
            self.t += dt

            if not self.phi_start:
                
                self.phi_start = True

            total_density = self.phi.density_vacuum() + self.phi.density_dark_radiation()
            # total_pressure = self.phi.pressure_vacuum() + self.phi.pressure_dark_radiation()
            
            
            pi_rho = (8 * np.pi * self.parameter['G'] / 3) * total_density
            pi_rho = max(0, pi_rho)
            self.a_dot = np.sqrt(pi_rho) * self.a

            self.a += self.a_dot * dt
            self.H = self.a_dot / self.a
            
            
            if not self.a > 0:
                self.destructor()
                break
            if not self.a_dot > 0:
                self.destructor()
                break


            self.phi.evolve(dt, self.a, self.a_dot)

            self.recorder['t'].append(self.t)
            self.recorder['a'].append(self.a)
            self.recorder['a_dot'].append(self.a_dot)
            self.recorder['H'].append(self.H)

        self.destructor()
        
    def destructor(self):
        for k, v in self.recorder.items():
            self.recorder[k] = np.array(v)
        for k, v in self.phi.recorder.items():
            self.phi.recorder[k] = np.array(v)


class Phi(Parameter, InitialValue):
    def __init__(self, parameter, ivalue) -> None:
        Parameter.__init__(self, parameter)
        InitialValue.__init__(self, ivalue)
        self.field = self.ivalue['phi']
        self.field_dot = self.ivalue['phi_dot']
        self.density_DR = self.ivalue['density_DR']
        self.alpha = self.parameter['g']**2 / (4 * np.pi)
        self.update_temperature()
        self.recorder = {'t': [], 
                         'field': [], 'field_dot': [], 'field_dot_dot': [],
                         'density_vacuum': [], 'density_DR': [], 
                         'T': [], 'Upsilon': []}
        self.t = self.ivalue['t']

        self.TF = False


    def update_temperature(self):
        self.T = (self.density_DR /
                  (2 * self.parameter['Nc']**2 - 1) / (np.pi**2 / 30))**(1/4)

    def evolve(self, dt, a, a_dot):
        self.t += dt

        H = a_dot / a
        self.density_DR_dot = -4 * H * self.density_DR + self.get_Upsilon() * self.field_dot**2
        self.field_dot_dot = -3 * H * self.field_dot - \
            self.get_Upsilon() * self.field_dot + self.parameter['C']
        
        if not self.TF:
            print(-3 * H * self.field_dot)
            print(-self.get_Upsilon() * self.field_dot)
            print(self.parameter['C'])
            self.TF = True

        self.density_DR += self.density_DR_dot * dt
        self.field_dot += self.field_dot_dot * dt
        self.field += self.field_dot * dt
        self.update_temperature()

        self.recorder['t'].append(self.t)
        self.recorder['field'].append(self.field)
        self.recorder['field_dot'].append(self.field_dot)
        self.recorder['field_dot_dot'].append(self.field_dot_dot)
        self.recorder['density_vacuum'].append(self.density_vacuum())
        self.recorder['density_DR'].append(self.density_dark_radiation())
        self.recorder['T'].append(self.T)
        self.recorder['Upsilon'].append(self.get_Upsilon())

    def get_Gamma_sph(self): # maybe need modification
        return self.parameter['Nc']**5 * self.alpha**5 * self.T**4

    def get_Upsilon(self):
        return self.get_Gamma_sph() / (2 * self.T * self.parameter['f']**2)

    def density_vacuum(self):
        density = (1/2) * self.field_dot**2 + self.potential()
        if density < 0:
            return 0
        return density

    def density_dark_radiation(self):
        return self.density_DR

    def pressure_vacuum(self):
        return - self.density_vacuum()

    def pressure_dark_radiation(self):
        return self.density_dark_radiation() / 3

    def potential(self):
        return - self.parameter['C'] * self.field

    def pressure(self):
        return (1/2) * self.field_dot**2 - self.potential()


if __name__ == '__main__':

    universe = Universe('Parameter.yaml', 'InitialValue.yaml')
    # universe.run(dt=-1e-7, iteration=26500)
    universe.run(dt=1e-5, iteration=2000)
