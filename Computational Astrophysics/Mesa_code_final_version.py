#Computational Astrophysics - Final Project
#Kaya Han Taş - 15064735 (Astronomy & Astrophysics)

#Project: Modelling Stellar Energy Transport (mini-MESA)

################################### SETUP #####################################

#We import os to change the directory of the Python's working directory to
#where opacity.txt is to avoid writing filepath.
import os 
os.chdir(r"C:\Users\kayah\OneDrive\Desktop\Astronomy and Astrophysics\Semester 2\Period 5\Computational Astrophysics\Final Project")

#We import warnings module since "interp2d" gives an error and we don't want
#it to be printed.
import warnings

############################ IMPORTING LIBRARIES ##############################

#We import the Libraries we need.
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

#We also import scipy constants and also astropy constants & units to define
#the constants and the Initial Conditions for our calculations.
import scipy.constants as const
from astropy import constants as astroconstant

#We import time and sys for printing Progress during the ODE solving.
import time
import sys

################################ DEFINITIONS ##################################

#We create a class to put everything into.
#Note that from now on all the definitions (including the functions) will be
#inside the class "Star_model"!
class Star_model:
    #__init__ function has everything that will be executed when the class we
    #defined is called.
    #We use it to assign values or other operations that are NECESSARY to do
    #when we create an object from it!
    def __init__(self):
        #We define some constants.
        self.boltzmann_constant = const.Boltzmann #SI Units
        self.ste_boltz_constant = const.Stefan_Boltzmann #SI Units
        self.c = const.speed_of_light #SI Units (m/sec) (Speed of Light)
        self.G = const.gravitational_constant  #SI Units (m^3/kg s^2) (Gravitational Constant)
        
        #We define the Solar Mass, Radius and Luminosity.
        self.L_sun = astroconstant.L_sun.value
        self.R_sun = astroconstant.R_sun.value
        self.M_sun = astroconstant.M_sun.value
 
        #Mean Molecular Weight (Mu) is given as well.
        X = 0.7
        Y = 0.29 + 1e-10
        Z = 1 - (X + Y)

        self.mu = 1 / (2*X + (3/4)*Y + (1/2)*Z)

        #We can also write Atomic Mass Unit as follows.
        self.mass_u = const.atomic_mass #SI Units (kg) 

        #We define the Initial Parameters as follows.
        #Best Fit for the Goals:
            #L x 1.1
            #R x 0.95
            #M x 1.0
            #T x 0.7
            #rho x 49
        self.L_initial = 1.0 * self.L_sun #SI Units
        self.R_initial = 1.0 * self.R_sun #SI Units (m)
        self.M_initial = 1.0 * self.M_sun #SI Units (kg)
        self.T_initial = 1.0 * 5770 #Kelvin

        #We also define the Average Density and calculate the Initial Density
        #by using that.
        self.rho_avg = 1.408e3 #SI Units (kg/m^3)
        self.rho_initial = 1.0 * 1.42e-7 * self.rho_avg #SI Units
        
        #We also define Heat Capacity which is needed on some calculations for
        #Temperature Gradient (Change in Temperature with respect to the Radial
        #Coordinate)
        self.Cp = (5/2) * (self.boltzmann_constant/(self.mu * self.mass_u))
        
        #We define end_step, which is going to store the final integration step
        #which will be used for plotting purposes.
        self.end_step = 0
        
        #We define adaptive_timestep.
        #If its True, we will use Adaptive Timestepping on our ODE Solving.
        #If its False, we won't use Adaptive Timestepping on our ODE Solving.
        self.adaptive_timestep = True

################################## TASK 1 #####################################
#First let's introduce Task 1.

#We first need to create a method that reads the file "opacity.txt".
    #We need to take Temperature (T) and Density (rho) as an input.
    #The method/function will return Opacity (kappa).
    
    #Note: The Temperature, Density and Opacity must be in SI units!
    
    #Note: We plan to use 2D Linear Interpolation for the common case where the
    #input value is not exactly found in the Opacity Table.
    
    #Note: We also plan to Extrapolate if we are outside of the Bounds of the
    #table, but we'll output a warning for it.
    
#The Structure of the "opacity.txt" file:
    #Top Row: log10(R) with R = rho/(T/10^6)^3. => rho = [gr/cm^3]
    #First Column: log10(T) => T = [K]
    #Rest of the Table: log10(kappa) => kappa = [cm^2/gr]

    #Now we create a function that does the things we have discussed.
    
    ###########################################################################    
    #NOTE: We put "self" first in order for our class to notice that the
    #function we define is its' attribute!!!
    #NOTE: We also put "self" in order to use the parameters we have defined
    #on the __init__(self) part of our class! (check Task 3 for an example)
    ###########################################################################
        
    def get_opacity(self, T, rho, log_R_given = [], log_T_given = [], expected_kappa = []):
        """Takes the Temperature and Density value. Returns the Opacity value 
        for that Temperature and Density by using the "opacity.txt" file.
        
        Input:
        ---------------------------------------------------------------------------
        T: Temperature in units of [Kelvin].
        rho: Density in units of [g/cm^3].
        log_R_given: A list that contains log(R) values. (Optional, for sanity 
        check)
        log_T_given: A list that contains log(T) values. (Optional, for sanity
        check)
        expected_kappa: A list that Contains Opacity with SI Units times 10^3 
        values. (Optional, for sanity check.)
        
        Output:
        ---------------------------------------------------------------------------
        kappa: Opacity in units of [m^2/kg].
        
        Notes:
        ---------------------------------------------------------------------------
        The function will use the Temperature and Density values to interpolate
        the Opacity value. In case of the Temperature and Density value being 
        out of bounds of the "opacity.txt" file, a warning will be printed out.
        
        """
        
        #We first open the file.
        data = np.genfromtxt("opacity.txt")
        
        #We get the Radius, Temperature and Opacity values from the file.
        #Note: For Radius and Temperature, we skip the first column of first 
        #row since it is Not a Number (NaN)!
        log10_R = data[0][1:]
        log10_T = data[1 : , 0]
        log10_kappa = data[ 1 : , 1 : ]
        
        #We calculate the Radius with the given Temperature and Density.
        #Note that since the File has Radius in log, we also take the log of the
        #calculation.
        #IMPORTANT: Since rho is in SI and in the table it is in cgs, we have
        #to turn our rho value into cgs by multiplying it with 10^-3!
        #We also check if log_R_given is empty or not.
        #If its empty, we will calculate log(R), if its full we'll use those
        #values.
        if log_R_given != [] and expected_kappa != []:
            R_given = log_R_given
        
        else:
            R_given = np.log10(rho * 0.001 / ((T/1e6)**3))
        
        #We need to check if our Temperature and Density is within the boundaries 
        #of the values we have on "opacity.txt".
        #For this, we do the following:
        if ((np.log10(T) < log10_T.min()) or (np.log10(T) > log10_T.max()) or
            (R_given < log10_R.min()) or (R_given > log10_R.max())):
            print(90 * "-")
            print("Warning: Opacity value is Extrapolated!")
            print(90 * "-")
        
        #Note: There is a warning message that appears for interp2d, to avoid that
        #we use warnings module.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            #We create a 2D Interpolation Function to Interpolate & Extrapolate.
            #For this we give the x, y and z (value we want to interpolate) to 
            #interp2d() function!
            #Note that we do Linear Interpolation here.
            interpolation_function = interp2d(log10_R, log10_T, log10_kappa, kind = "linear")
        
            #Now we interpolate the Opacity value as follows.
            #Note that since the Opacity values are also in log, we need to revert 
            #it back to normal.
            if log_T_given != [] and expected_kappa != []:
                print(90 * "-")
                print("Opacity Sanity Check")
                for i in range(len(R_given)):
                    kappa = interpolation_function(log_R_given[i], log_T_given[i])[0]
                    #We turn it from log to normal value, then turn it into SI
                    #and then multiply it by 10^3 to compare it to the table.
                    kappa = 10 ** kappa * 1e-1 * 1e3
                    print(90 * "-")
                    print(f"log(T) = {log_T_given[i]}")
                    print(f"log(R) = {log_R_given[i]}")
                    print(f"Opacity (Interpolated) (SI)= {kappa}")
                    print(f"Opacity (Expected) (SI)= {expected_kappa[i]}")
                    print(f"Relative Error: {((np.abs(expected_kappa[i] - kappa)/np.abs(expected_kappa[i])) * 100):.5f} %")
                    if ((np.abs(expected_kappa[i] - kappa)/np.abs(expected_kappa[i])) * 100) < 5:
                        print("Interpolation works well.")
                    
                    else:
                        print("Warning: Interpolation has some error.")
                        
                    print(90 * "-")
            
            else:
                kappa = interpolation_function(R_given, np.log10(T))[0]
            
            kappa = 10 ** kappa
            
            #Since kappa is an array due to the output from the interp2d, we take
            #it out of the array to get a value.
            #Note that 10^-1 comes from conversion from cgs to SI!
            kappa = kappa * 1e-1
        
        #We return the Opacity value.
        return kappa

################################## TASK 2 #####################################
#First let's introduce Task 2.

#Our plan is to implement a similar method for "epsilon.txt".
#The structure of it is same as the "opacity.txt" opacity table.

    #Now we create a function that does the things we have discussed.
    def get_total_energy(self, T, rho, log_R_given = [], log_T_given = [], expected_epsilon = []):
        """Takes the Temperature and Density value. Returns the Total Energy 
        value for that Temperature and Density by using the "epsilon.txt" file.
        
        Input:
        ---------------------------------------------------------------------------
        T: Temperature in units of [Kelvin].
        rho: Density in units of [gr/cm^3].
        log_R_given: A list that contains log(R) values. (Optional, only for 
        sanity check)
        log_T_given: A list that contains log(T) values. (Optional, only for 
        sanity check)
        expected_epsilon: A list that Contains Get Total Energy with SI Units 
        times 10^92 values. (Optional, only for sanity check)
        
        Output:
        ---------------------------------------------------------------------------
        epsilon: Total Energy Released from Fusion Reactions in the Core of the
        Star in terms of SI Units.
        
        Notes:
        ---------------------------------------------------------------------------
        The function will use the Temperature and Density values to interpolate
        the Total Energy value. In case of the Temperature and Density value 
        being out of bounds of the "epsilon.txt" file, a warning will be 
        printed out.
        
        """
        
        #We first open the file.
        data = np.genfromtxt("epsilon.txt")
        
        #We get the Radius, Temperature and Total Energy values from the file.
        #Note: For Radius and Temperature, we skip the first column of first row 
        #since it is Not a Number (NaN)!
        log10_R = data[0][1:]
        log10_T = data[1 : , 0]
        log10_epsilon = data[ 1 : , 1 : ]
        
        #We calculate the Radius with the given Temperature and Density.
        #Note that since the File has Radius in log, we also take the log of the
        #calculation.
        #IMPORTANT: Since rho is in SI and in the table it is in cgs, we have
        #to turn our rho value into cgs by multiplying it with 10^-3!
        if log_R_given != [] and expected_epsilon != []:
            R_given = log_R_given
        
        else:
            R_given = np.log10(rho * 0.001 / ((T/1e6)**3))
        
        #We need to check if our Temperature and Density is within the boundaries 
        #of the values we have on "epsilon.txt".
        #For this, we do the following:
        if ((np.log10(T) < log10_T.min()) or (np.log10(T) > log10_T.max()) or
            (R_given < log10_R.min()) or (R_given > log10_R.max())):
            print(90 * "-")
            print("Warning: Total Energy value is Extrapolated!")
            print(90 * "-")
        
        #Note: There is a warning message that appears for interp2d, to avoid that
        #we use warnings module.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            #We create a 2D Interpolation Function to Interpolate & Extrapolate.
            #For this we give the x, y and z (value we want to interpolate) to 
            #interp2d() function!
            #Note that we do Linear Interpolation here.
            interpolation_function = interp2d(log10_R, log10_T, log10_epsilon, kind = "linear")
        
            #Now we interpolate the Total Energy value as follows.
            #Note that since the Total Energy values are also in log, we need to revert 
            #it back to normal.
            if log_T_given != [] and expected_epsilon != []:
                print(90 * "-")
                print("Total Generated Energy Sanity Check")
                for i in range(len(R_given)):
                    epsilon = interpolation_function(log_R_given[i], log_T_given[i])[0]
                    #We turn it from log to normal value, then turn it into SI
                    #and then multiply it by 10^92 to compare it to the table.
                    epsilon = 10 ** epsilon * 1e-4 * 1e92
                    print(90 * "-")
                    print(f"log(T) = {log_T_given[i]}")
                    print(f"log(R) = {log_R_given[i]}")
                    print(f"Total Energy (Interpolated) (SI)= {epsilon}")
                    print(f"Total Energy (Expected) (SI)= {expected_epsilon[i]}")
                    print(f"Relative Error: {((np.abs(expected_epsilon[i] - epsilon)/np.abs(expected_epsilon[i])) * 100):.5f} %")
                    if ((np.abs(expected_epsilon[i] - epsilon)/np.abs(expected_epsilon[i])) * 100) < 5:
                        print("Interpolation works well.")
                    
                    else:
                        print("Warning: Interpolation has some error.")
                        
                    print(90 * "-")
            
            else:
                epsilon = interpolation_function(R_given, np.log10(T))[0]
            
            epsilon = 10 ** epsilon
            
            #Since epsilon is an array due to the output from the interp2d, we take
            #it out of the array to get a value.
            #Note that 10^-4 comes from conversion from cgs to SI!
            epsilon = epsilon * 1e-4
        
        #We return the Total Energy value. (in terms of SI Units)
        return epsilon

################################## TASK 3 #####################################
#First let's introduce Task 3.

#Our goal is to calculate: 
    #Density given Pressure and Temperature => Equation 6 on the Paper!
    #Pressure given Density and Temperature => Equation 5 on the Paper!

    #First we define a function to calculate the Density!
    def density_calculator(self, P, T):
        """Takes the Pressure and Temperature value. Calculates the Density and
        returns it.
        
        Input:
        ---------------------------------------------------------------------------
        P: Pressure in units of [N/m^2] (SI Unit)
        T: Temperature in units of [Kelvin].
        
        Output:
        ---------------------------------------------------------------------------
        rho: Density in units of [kg/m^3].
        
        """
        
        #We use Equation 6 from the paper.
        #We divide the equation into two parts for the sake of understandable code.
        eq1 = (P - ((4 * self.ste_boltz_constant * (T**4))/(3 * self.c)))
        eq2 = (self.mu * self.mass_u) / (self.boltzmann_constant * T)
        rho = eq1 * eq2
        
        return rho
    
    #Now we define a function to calculate the Pressure!
    def pressure_calculator(self, rho, T):
        """Takes the Density and Temperature value. Calculates the Pressure and
        returns it.
        
        Input:
        ---------------------------------------------------------------------------
        rho: Density in units of [kg/m^3].
        T: Temperature in units of [Kelvin].
        
        Output:
        ---------------------------------------------------------------------------
        P: Pressure in units of [N/m^2] (SI Unit)
        
        """
        
        #We use Equation 6 from the paper.
        #We divide the equation into two parts for the sake of understandable code.
        eq1 = ((4 * self.ste_boltz_constant)/(3 * self.c)) * (T ** 4)
        eq2 = ((rho * self.boltzmann_constant)/(self.mu * self.mass_u)) * T
        P = eq1 + eq2
        
        return P

################################## TASK 4 #####################################
#First let's introduce Task 4.

#Our goal is to solve 4 Partial Differential Equations (PDEs) for Stellar
#Structure numerically.
#For this we'll write our own ODE solvers. (Euler or Runge Kutta or Simpson's)
#We plan to include Adaptive Timestepping.
#For Interpolation and Root Finding we will use packages.
#We also need to check the change in Temperature with respect to the Mass
#coordinate by checking the Stable and Adiabatic Temperature Gradients, which
#will be implemented in the function we will define.

    #Now we define our first function for Change in Radial Coordinate.
    def radial_coordinate_change(self, r, rho):
        """Takes the Mass Coordinate and Density. Returns the Change in Radial 
        Coordinate with respect to the Mass Coordinate.
        
        Input:
        ---------------------------------------------------------------------------
        m: Mass Coordinate
        rho: Density (In SI Units i.e. kg/m^3)
        
        Output:
        ---------------------------------------------------------------------------
        dr_dm: Change in Radial Coordinate with respect to the Mass Coordinate.
        
        """
        #We use the Stellar Structure equation. => Equation 1 on the Paper!
        dr_dm = 1 / (4 * np.pi * r**2 * rho)
        
        return dr_dm
    
    #We now define a function for Change in Pressure.
    def pressure_change(self, r, m):
        """Takes the Radial Coordinate and Mass Coordinate. Returns the change 
        in Pressure with respect to the Mass Coordinate.
        
        Input:
        ---------------------------------------------------------------------------
        r: Radial Coordinate
        m: Mass Coordinate
        
        Output:
        ---------------------------------------------------------------------------
        dP_dm: Change in Pressure with respect to the Mass Coordinate.
        
        """
        
        #We use the Stellar Structure Equation. => Equation 2 on the Paper!
        dP_dm = -self.G * m/(4 * np.pi * r**4)
        
        return dP_dm
    
    #We now define the Change in Luminosity.
    def luminosity_change(self, T, rho):
        """Takes the Temperature and Density. Returns the change in Luminosity with 
        respect to the Mass Coordinate by using the previously defined 
        get_total_energy function which finds the corresonding Total Energy from 
        the "epsilon.txt" file for given Temperature and Density.
        
        Input:
        ---------------------------------------------------------------------------
        T: Temperature in units of [Kelvin].
        rho: Density in units of [g/cm^3].
        
        Output:
        ---------------------------------------------------------------------------
        dL_dm: Change in Luminosity with respect to the Mass Coordinate.
        
        """
        
        #We first use our get_total_energy(T, rho) function to get the Total 
        #Energy for the given Temperature and Density.
        #Note that since "get_total_energy" function is defined inside our 
        #class, we have to use self.get_total_energy to call it!
        epsilon = self.get_total_energy(T, rho)
        
        #Since the Change in Luminosity is equal to the Total Energy we can find
        #dL/dm as follows.
        dL_dm = epsilon
        
        return dL_dm

    #Now we define our fourth and final function for change in Temperature.
    def temperature_change(self, r, rho, T, L):
        """Takes Radial Coordinate, Density, Temperature and Luminosity. 
        Returns the change in Temperature with respect to the Mass Coordinate.

        Input:
        ---------------------------------------------------------------------------
        r: Radial Coordinate.
        rho: Density in units of [g/cm^3].
        T: Temperature in units of [Kelvin].
        L: Luminosity in units of [SI Units].
        
        Output:
        ---------------------------------------------------------------------------
        dT_dm: Change in Temperature with respect to the Mass Coordinate.
        
        """
        
        #We can calculate Change in Temperature as follows.
        #Again we separate the equation into two and combine them.
        dT_dm1 = -(3 * self.get_opacity(T, rho) * L)
        dT_dm2 = (256 * np.pi**2 * self.ste_boltz_constant * r**4 * T**3)
        dT_dm = dT_dm1/dT_dm2 
        
        return dT_dm
    
    #For Temperature change we have to consider the Temperature Gradients as
    #well.
        #Stable Temperature Gradient (nabla_stable_calculate)
        #Adiabatic Temperature Gradient (nabla_ad_calculate)
        #Convective Temperature Gradient (nabla_star_calculate)
    
    #To calculate the Stable Temperature Gradient we define the following.
    def nabla_stable_calculate(self, m, r, rho, T, L, Hp, P):
        """Takes Radial Coordinate, Density, Temperature, Luminosity and Scale
        Height. Returns the Stable Temperature Gradient.

        Input:
        ---------------------------------------------------------------------------
        r: Radial Coordinate.
        rho: Density in units of [g/cm^3].
        T: Temperature in units of [Kelvin].
        L: Luminosity in units of [SI Units].
        Hp: Scale Height.
        
        Output:
        ---------------------------------------------------------------------------
        nabla_stable: Stable Temperature Gradient.
        
        Note:
        ---------------------------------------------------------------------------
        To calculate the Stable Temperature Gradient we use Scale Height (Hp) 
        which we will calculate outside of the function.
        
        """
        
        #We calculate Stable Temperature Gradient. => Equation 5.18 on Onno Pols!
        #nabla_stable = 3 * self.get_opacity(T, rho) * L * P / (64 * np.pi * self.ste_boltz_constant * self.G * m * T**4)
        
        #Equation 7 on the Paper but it doesn't work.
        #Unless you add a rho to it, in that case it works.
        nabla_stable = (3 * self.get_opacity(T, rho) * Hp * L * rho)/(64 * np.pi * r**2 * self.ste_boltz_constant * T**4)
        
        return nabla_stable
    
    #To calculate the Adiabatic Temperature Gradient we define the following.
    def nabla_ad_calculate(self, rho, P, T):
        """Takes Density, Pressure and Temperature. Returns the Adiabatic 
        Temperature Gradient.

        Input:
        ---------------------------------------------------------------------------
        rho: Density in units of [g/cm^3].
        P: Pressure in units of [SI Units].
        T: Temperature in units of [Kelvin].
        
        Output:
        ---------------------------------------------------------------------------
        nabla_ad: Adiabatic Temperature Gradient.
        
        Note:
        ---------------------------------------------------------------------------
        To calculate the Adiabatic Temperature Gradient we use the Heat 
        Capacity we have defined on our class as self.Cp.
        
        """
        
        #We calculate Adiabatic Temperature Gradient. => Equation 10 on the Paper!
        nabla_ad = P / (T * rho * self.Cp)
        
        return nabla_ad
    
    #To calculate the Convective Temperature Gradient we define the following.
    def nabla_star_calculate(self, rho, T, Hp, g, nabla_stable, nabla_ad):
        """Takes Density, Pressure and Temperature. Returns the Adiabatic 
        Temperature Gradient.

        Input:
        ---------------------------------------------------------------------------
        rho: Density in units of [g/cm^3].
        T: Temperature in units of [Kelvin].
        Hp: Scale Height.
        g: Gravitational Acceleration at given Mass Shell.
        nabla_stable: Stable Temperature Gradient.
        nabla_ad: Adiabatic Temperature Gradient.
        
        Output:
        ---------------------------------------------------------------------------
        nabla: Convective Temperature Gradient. (Which is equal to THE
        Temperature Gradient in the case of nabla_ad < nabla_stable.)
        
        Note:
        ---------------------------------------------------------------------------
        To calculate the Convective Temperature Gradient we use the Scale
        Height (Hp) and Gravitational Acceleration (g) which we will calculate
        outside of this function.
        
        """
        
        #We first calculate U. (since its a long equation we write it separated
        #then multiply them.) => Equation 12.a on the Paper!
        U1 = (64 * self.ste_boltz_constant * T**3) / (3 * self.get_opacity(T, rho) * rho**2 * self.Cp)
        U2 = np.sqrt(Hp/g)
        U = U1 * U2
        
        #Now we calculate lm (mixing length) which is equal to the Scale 
        #Height "Hp".
        lm = Hp
        
        #Finally, we calculate Omega as follows.
        Omega = 4/lm
        
        #Now we have to find the roots of the 3rd Degree Polynomial given on
        #the paper as Equation 12 in order to find Convective Temperature
        #Gradient!
        
        #We first define the coefficients of the Polynomial.
        coefficients = [1,
                        U/lm**2, 
                        U**2/lm**3 * Omega, 
                        U / lm**2 * (nabla_ad - nabla_stable)]
        
        #Now we find the roots of the Cubic Equation as follows.
        roots = np.roots(coefficients)
        
        #We need the real roots from the equation, hence we do the following.
        #np.isreal(roots): Returns boolean that tells whether the root is
        #real or not.
        #.real: Filters Real roots and turns them into real number format.
        real_root = roots[np.isreal(roots)].real
        
        #Now we use the Root of the 3rd Degree Polynomial to find the
        #Convective Temperature Gradient as follows.
        #Note that we take the 0th index of the real_root since the output from
        #root finding is in a Numpy array.
        real_root = real_root[0]
        nabla_star = real_root ** 2 + (((U * Omega)/lm) * real_root) + nabla_ad
        
        #In this case, the Convective Temperature Gradient is equal to THE
        #Temperature Gradient.
        nabla = nabla_star
        
        return nabla
    
    
    #We also define some functions for the Flux Calculations since it will be
    #required to do the Cross-Section of the Star to see if its Radiative or
    #Convective.
    
    #We first define the Total Flux.
    def total_flux(self, r, L):
        """Takes Radial Coordinate and Luminosity. Returns the Total Flux.

        Input:
        ---------------------------------------------------------------------------
        r: Radial Coordinate.
        L: Luminosity in units of [SI Units].
        
        Output:
        ---------------------------------------------------------------------------
        flux: Total Flux.

        """
        
        #We calculate the Total Flux.
        flux = L/(4 * np.pi * r**2)
        
        return flux
    
    #We then define the Radiative Flux.
    def radiative_flux(self, rho, T, kappa, Hp, nabla):
        """Takes Density, Temperature, the Opacity, Scale Height and THE
        Temperature Gradient. Returns the Radiative Flux.
        
        Input:
        ---------------------------------------------------------------------------
        rho: Density in units of [g/cm^3].
        T: Temperature in units of [Kelvin].
        kappa: Opacity in units of [m^2/kg].
        Hp: Scale Height.
        nabla: THE Temperature Gradient.
        
        Output:
        ---------------------------------------------------------------------------
        rad_flux: Radiative Flux.
        
        Note:
        ---------------------------------------------------------------------------
        To calculate the Radiative Flux we need Scale Height (Hp) and THE
        Temperature Gradient (nabla) which we will calculate outside of this
        function.

        """
        
        #We calculate the Radiative Flux.
        rad_flux = 16 * self.ste_boltz_constant * T**4 / (3 * kappa * rho * Hp) * nabla
        
        return rad_flux
    
    #We finally define the Convective Flux.
    def convective_flux(self, total_flux, radiative_flux):
        """Takes the Total Flux and Radiative Flux. Returns the Convective Flux
        assuming no Conduction.
        
        Input:
        ---------------------------------------------------------------------------
        total_flux: Total Flux.
        radiative_flux: Radiative Flux.
        
        Output:
        ---------------------------------------------------------------------------
        con_flux: Convective Flux.
        
        """
        
        #We define the Convective Flux by substracting the Radiative Flux from
        #the Total Flux.
        #This is reasonable since we assume there is no Conduction.
        con_flux = total_flux - radiative_flux
        
        return con_flux
    
    #Now we have to solve the functions per time step.
    #For this we will use Euler method.
    #We will also include Adaptive Timestepping as well.
    #For this we create the following function.
    
    def ODE_solver(self):
        """Returns Radius, Pressure, Temperature, Luminosity, Density, Total
        Generated Energy, Mass, THE Temperature Gradient, Convective 
        Temperature Gradient, Stable Temperature Gradient, Adiabatic 
        Temperature Gradient, Convective Flux, Radiative Flux and Total Flux 
        values per Integration Step. Uses Euler Method to do the Integration.
        
        Input:
        ---------------------------------------------------------------------------
        None.
        
        Output:
        ---------------------------------------------------------------------------
        r: Numpy Array containing Radius values per integration step.
        P: Numpy Array containing Pressure values per integration step.
        L: Numpy Array containing Luminosity values per integration step.
        T: Numpy Array containing Temperature values per integration step.
        rho: Numpy Array containing Density values per integration step.
        epsilon: Numpy Array containing Total Generated Energy values per 
        integration step.
        m: Numpy Array containing Mass values per integration step.
        nabla: Numpy Array containing THE Temperature Gradient values per 
        integration step.
        nabla_star: Numpy Array containing Convective Temperature Gradient 
        values per integration step.
        nabla_stable: Numpy Array containing Stable Temperature Gradient 
        values per integration step.
        nabla_ad: Numpy Array containing Adiabatic Temperature Gradient 
        values per integration step.
        FC: Numpy Array containing Convective Flux values per integration step.
        FR: Numpy Array containing Radiative Flux values per integration step.
        F: Numpy Array containing Total Flux values per integration step.
        
        """
        
        #We first start a timer to check how long the solving takes.
        start_time = time.time()
        
        #We define the Number of Steps we will take.
        num_step = int(1e5)
        
        #We define the numpy arrays to store the parameters we want to check.
        #Note that we define the arrays so that it is made out of zeros with
        #length being the number of steps we will take.
        
        #Main Parameters
        m = np.zeros(num_step)
        r = np.zeros(num_step)
        P = np.zeros(num_step)
        L = np.zeros(num_step)
        T = np.zeros(num_step)
        rho = np.zeros(num_step)
        epsilon = np.zeros(num_step)
        
        #The Temperature Gradients
        nabla = np.zeros(num_step)
        nabla_star = np.zeros(num_step)
        nabla_stable = np.zeros(num_step)
        nabla_ad = np.zeros(num_step)
        
        #The Fluxes
        F = np.zeros(num_step)
        FR = np.zeros(num_step)
        FC = np.zeros(num_step)
        
        #We define the Initial Values of the Main Parameters by using the
        #variables inside __init__ we have defined on our Star_model() class.
        m[0] = self.M_initial
        r[0] = self.R_initial
        P[0] = self.pressure_calculator(self.rho_initial, self.T_initial)
        L[0] = self.L_initial
        T[0] = self.T_initial
        rho[0] = self.rho_initial
        epsilon[0] = self.get_total_energy(self.T_initial, self.rho_initial)
        
        #We can also define the Initial Values of the Temperature Gradients by
        #using the functions we have defined for them.
        
        #We first calculate the initial Gravitational Acceleration.
        g = self.G * m[0] / r[0] ** 2
        
        #Then we calculate the initial Scale Height.
        Hp = self.boltzmann_constant * T[0]/ (self.mu * self.mass_u * g)
        
        #We can now calculate the initial Stable Temperature Gradient.
        nabla_stable[0] = self.nabla_stable_calculate(m[0], r[0], rho[0], T[0], L[0], Hp, P[0])
        
        #We can also calculate the initial Adiabatic Temperature Gradient now.
        nabla_ad[0] = self.nabla_ad_calculate(rho[0], P[0], T[0])
        
        #We also calculate the initial Convective Temperature Gradient.
        nabla_star[0] = self.nabla_star_calculate(rho[0], T[0], Hp, g, nabla_stable[0], nabla_ad[0])
        
        #We can now do steps to solve our Stellar Structure Equations.
        for i in range(num_step - 1):
            #We use sys.stdout.write to write into the Python Console.
            #We define a Progress Percentage where for each step we will update
            #it by checking the current mass shell we are on, divide it by the
            #initial mass to get a precentage.
            #We update it every 20 steps so that it doesn't take too much space
            #in the console.
            if i % 20 == 0:
                #print("Progress: %3.4f %s\r" % (100 - (m[i] / self.M_initial) * 100, "%"))
                sys.stdout.write("Progress: %3.4f %s\r" % (100 - (m[i] / self.M_initial) * 100, "%"))
            
                #We use sys.stdout.flush() to update the console immediately as we
                #move through the mass shells.
                sys.stdout.flush()

            #We check if Radius, Luminosity, Density, Pressure or Temperature
            #is close to zero.
            #If they are close to zero, we end the steps and print out the 
            #values on the step we ended on.
            if r[i] < 1e-4 or L[i] < 1e-4 or rho[i] < 1e-4 or P[i] < 1e-4 or T[i] < 1e-4:
                print(90 * "-")
                print("Parameter other than Mass has reached zero!")
                print(90 * "-")
                print("Step:", i)
                print(90 * "-")
                print("Mass:", m[i])
                print(f"Equal to: {m[i]*100/self.M_initial:.2f}% of Initial Mass")
                print(90 * "-")
                print("Radius:", r[i])
                print(f"Equal to: {r[i]*100/self.R_initial:.2f}% of Initial Radius")
                print(90 * "-")
                print("Luminosity:", L[i])
                print(f"Equal to: {L[i]*100/self.L_initial:.2f}% of Initial Luminosity")
                print(90 * "-")
                print("Density:", rho[i])
                print(90 * "-")
                print("Pressure:", P[i])
                print(90 * "-")
                print("Temperature:", T[i])
                print(90 * "-")
                
                #We get the ending step number in order to use it for plotting 
                #later on.
                #We subtract 1 since in the last step one of the parameters has 
                #hit zero (or negative) which may cause issues during plotting.
                self.end_step = i - 1
                
                #We stop the loop.
                break
            
            #We check if the Mass is close to zero.
            #If it is close to zero, it means that we have reached the core
            #for which we can end the loop.
            elif m[i] < 1e-4:
                print(90 * "-")
                print("Mass has reached zero!")
                print(90 * "-")
                print("Final values:")
                print(90 * "~")
                print(90 * "~")
                print("Step:", i)
                print(90 * "-")
                print("Mass:", m[i])
                print(90 * "-")
                print("Radius:", r[i])
                print(f"{r[i]*100/self.R_initial:.2f}% of Initial Radius")
                print(90 * "-")
                print("Luminosity:", L[i])
                print(f"{L[i]*100/self.L_initial:.2f}% of Initial Luminosity")
                print(90 * "-")
                print("Density:", rho[i])
                print(90 * "-")
                print("Pressure:", P[i])
                print(90 * "-")
                print("Temperature:", T[i])
                print(90 * "-")
                
                #We get the ending step number in order to use it for plotting 
                #later on.
                #We subtract 1 since in the last step mass has hit zero (or
                #negative) which may cause issues during plotting.
                self.end_step = i - 1
                
                #We stop the loop.
                break
            
            else:
                #We calculate the Opacity in the shell we are on by using the 
                #get_opacity function.
                kappa = self.get_opacity(T[i], rho[i])
                
                #We also calculate the Gravitational Acceleration and Scale
                #Height at the shell we are on.
                g = self.G * m[i] / r[i] ** 2
                Hp = self.boltzmann_constant * T[i]/ (self.mu * self.mass_u * g)
                
                #We calculate the Stable, Adiabatic and Convective Temperature 
                #Gradient at the shell we are on.
                nabla_stable[i] = self.nabla_stable_calculate(m[i], r[i], rho[i], T[i], L[i], Hp, P[i])
                nabla_ad[i] = self.nabla_ad_calculate(rho[i], P[i], T[i])
                nabla_star[i] = self.nabla_star_calculate(rho[i], T[i], Hp, g, nabla_stable[i], nabla_ad[i])
                
                #We calculate the Total Flux at the shell we are on.
                F[i] = self.total_flux(r[i], L[i])
                
                #If the Adiabatic Temperature Gradient is smaller than Stable
                #Temperature Gradient we do the following.
                if nabla_ad[i] < nabla_stable[i]:
                    #We calculate THE Temperature Gradient by using the 
                    #nabla_star function we have defined.
                    nabla[i] = nabla_star[i]
                    
                    #We calculate the Radiative and Convective Flux at that
                    #shell.
                    FR[i] = self.radiative_flux(rho[i], T[i], kappa, Hp, nabla[i])
                    FC[i] = self.convective_flux(F[i], FR[i])
                    
                    #We finally calculate the Change in Temperature.
                    # => Equation 4 on the Paper!
                    dT = nabla[i] * (T[i] / P[i]) * self.pressure_change(r[i], m[i])
                
                #If the Adiabatic Temperature Gradient is not smaller than 
                #Stable Temperature Gradient we do the following.
                else:
                    #THE Temperature Gradient is equal to the Stable 
                    #Temperature Gradient.
                    nabla[i] = nabla_stable[i]
                    
                    #The Radiative Flux is equal to the Total Flux.
                    FR[i] = F[i]
                    
                    #The Change in Temperature can be calculated using our
                    #defined function.
                    dT = self.temperature_change(r[i], rho[i], T[i], L[i])
                
                #We check if the adaptive_timestep in __init__ is True.
                #If it is, we do the Adaptive Timestepping.
                #For this method refer to: https://www.uio.no/studier/emner/matnat/astro/AST3310/v24/projects/project2/variablesteplength.pdf
                if self.adaptive_timestep:
                    #We define an array with dr/dm, dP/dm, dL/dm and dT/dm
                    #values and take their absolute i.e. turn all of them
                    #positive.
                    f = np.abs(np.array([self.radial_coordinate_change(r[i], rho[i]), 
                                         self.pressure_change(r[i], m[i]), 
                                         self.get_total_energy(T[i], rho[i]), 
                                         dT]))
                    
                    #We also create an array that contains the Radius, Pressure,
                    #Luminosity and Temperature of the current shell.
                    V = np.array([r[i], P[i], L[i], T[i]])
                    
                    #We define a small percentage to control step size going
                    #too high.
                    #Note that if p is large, the stepsize will be large and
                    #the results we obtain will have huge gaps between them.
                    p = 0.01
                    
                    #We calculate the dm value by dividing the Changes in
                    #parameters over dm with the parameters itself.
                    #Note that we determine the new dm (mass coordinate change) 
                    #as the minimum difference between f and V.
                    dm = np.min(p * V / f)
                    
                    #If dm is smaller than 10^19 the stepsize would be too
                    #small and it would continue to decrease which would cause
                    #the calculation to take too long.
                    #For that reason, we define it so that if dm is smaller
                    #than 10^19, we will consider it is 10^19.
                    if dm < 1e19:
                        dm = 1e19

                #We update the Arrays we have defined by using the results we
                #obtain from our functions and multiplying it with the dm so
                #that the dm terms cancel out. (Example: dr/dm * dm = dr!)
                r[i + 1] = r[i] - self.radial_coordinate_change(r[i], rho[i]) * dm
                P[i + 1] = P[i] - self.pressure_change(r[i], m[i]) * dm
                L[i + 1] = L[i] - self.get_total_energy(T[i], rho[i]) * dm    
                T[i + 1] = T[i] - dT * dm
                m[i + 1] = m[i] - dm
                rho[i + 1] = self.density_calculator(P[i+1], T[i+1])
                epsilon[i + 1] = self.get_total_energy(T[i+1], rho[i+1])

        print(f"Modelling is Finished. \n Time elapsed: {time.time() - start_time :.2f} seconds.")
        
        return r, P, L, T, rho, epsilon, m, nabla, nabla_star, nabla_stable, nabla_ad, FC, FR, F
    
    def plot_parameters(self, m_values, r_values, rho_values, P_values, T_values, L_values, nabla_star, nabla_stable, nabla_ad):
        """Takes the Mass, Radius, Density, Pressure, Temperature, Luminosity,
        Convective Temperature Gradient, Stable Temperature Gradient and
        Adiabatic Temperature Gradient values. Returns plots of the given
        arrays.
        
        Input:
        ---------------------------------------------------------------------------
        m_values: Numpy Array containing Mass values per step.
        r_values: Numpy Array containing Radius values per step.
        rho_values: Numpy Array containing Density values per step.
        P_values: Numpy Array containing Pressure values per step.
        T_values: Numpy Array containing Temperature values per step.
        L_values: Numpy Array containing Luminosity values per step.
        nabla_star: Numpy Array containing Convective Temperature Gradient 
        values per step.
        nabla_stable: Numpy Array containing Stable Temperature Gradient values
        per step.
        nabla_ad: Numpy Array containing Adiabatic Temperature Gradient values 
        per step.
        
        Output:
        ---------------------------------------------------------------------------
        Mass vs. Radius Plot
        Radius vs. Integration Step Plot
        Density vs. Radius Plot
        Pressure vs. Radius Plot
        Temperature vs. Radius Plot
        Luminosity vs. Radius Plot
        Temperature Gradient Comparison Plot
        
        Note:
        ---------------------------------------------------------------------------
        Density vs. Radius, Pressure vs. Radius and Temperature Gradient 
        Comparison plots are log-scaled on the y-axis while others are linear 
        scaled.
        
        """
        
        #We plot the Mass against Radius.
        plt.plot(r_values[0: self.end_step] / self.R_initial, m_values[0: self.end_step] / self.M_initial)
        plt.title("Mass vs. Radius")
        plt.xlabel(r"$r/R_0$ [m]")
        plt.ylabel(r"$m/M_0$ [kg]")
        plt.tight_layout()
        plt.show()
        
        #We plot the Radius against Integration Step "i".
        plt.plot(r_values[0: self.end_step]/r_values[0])
        plt.title("Radius vs. Integration Step")
        plt.xlabel("Integration Step (i)")
        plt.ylabel(r"$r/R_0$ [m]")
        plt.tight_layout()
        plt.show()
        
        #We plot Density against Radius.
        #Note: We use "symlog" here for the y scaling.
        #Symmetric Log Scale: Allows Positive and Negative Values! 
        #Log Scale: Only allows Positive values!
        plt.plot(r[0: self.end_step] / self.R_initial, rho[0: self.end_step] / self.rho_initial)
        plt.title("Density vs. Radius")
        plt.xlabel(r"$r/R_0$ [m]")
        plt.ylabel("$ρ/ρ_0$ [$kg/m^3$]")
        plt.yscale("symlog")
        plt.tight_layout()
        plt.show()
        
        #We plot Pressure against Radius.
        plt.plot(r[0: self.end_step] / self.R_initial, P[0: self.end_step] / P_values[0])
        plt.title("Pressure vs. Radius")
        plt.xlabel(r"$r/R_0$ [m]")
        plt.ylabel("$P/P_0$ [$N/m^2$]")
        plt.yscale("symlog")
        plt.tight_layout()
        plt.show()
        
        #We plot Temperature against Radius.
        plt.plot(r[0: self.end_step] / self.R_initial, T[0: self.end_step])
        plt.title("Temperature vs. Radius")
        plt.xlabel(r"$r/R_0$ [m]")
        plt.ylabel("$T/T_0$ [K]")
        plt.tight_layout()
        plt.show()
        
        #We plot Luminosity against Radius.
        plt.plot(r[0 : self.end_step] / self.R_initial, L[0 : self.end_step] / self.L_initial)
        plt.title("Luminosity vs. Radius")
        plt.xlabel(r"$r/R_0$ [m]")
        plt.ylabel("$L/L_0$ [W]")
        plt.tight_layout()
        plt.show()
        
        #We plot Stable, Convective and Adiabatic Temperature Gradients.
        plt.title("Temperature Gradients for $p = 0.01$")
        plt.xlabel("$\\frac{r}{R_{\\mathrm{sun}}}$")
        plt.ylabel("$\\nabla$")
        plt.plot(r[0 : self.end_step] / self.R_sun, nabla_stable[0 : self.end_step], label = "$\\nabla_{stable}$")
        plt.plot(r[0 : self.end_step] / self.R_sun, nabla_star[0 : self.end_step], label = "$\\nabla_{*}$") 
        plt.plot(r[0 : self.end_step] / self.R_sun, nabla_ad[0 : self.end_step], label = "$\\nabla_{ad}$")
        plt.yscale("symlog")
        #plt.ylim(0, 1e3)
        plt.legend()
        plt.show()
    
    #For the following Cross Section plotting please refer to the website below:
        #https://www.uio.no/studier/emner/matnat/astro/AST3310/v24/projects/project2/cross_section.py
    
    def plot_cross_section(self, r_values, L_values, FC_values):
        """Takes the Normalized Radius, Luminosity, and Convective Flux values. 
        Returns the Cross Section of the Star and its Regions.
        
        Input:
        ---------------------------------------------------------------------------
        r_values: Numpy Array containing Normalized Radius values per step.
        L_values: Numpy Array containing Normalized Luminosity values per step.
        FC_values: Numpy Array containing Normalized Convective Flux values per 
        step.
        
        Output:
        ---------------------------------------------------------------------------
        Cross Section of the Star plot.
        
        Note:
        ---------------------------------------------------------------------------
        To Normalize the Radius and Luminosity values, we need to divide it
        with the Solar Radius and Solar Luminosity unit.
        
        """
        #We define a figure.
        plt.figure(figsize=(7, 7))
        
        #We get the current axes by using plt.gca().
        ax = plt.gca()
        
        #We will need to check all the values inside our arrays one by one.
        #For that reason, we define a variable that is the length of our
        #arrays.
        n = len(r_values)
        
        #We define the Last Convective Step.
        #We do this in order to define the amount of Convective Layer Outside
        #Core.
        last_conv_step = 0
        
        #We define the First Convective Step.
        #We do this in order to define the amount of Convective Layer Inside
        #Core.
        first_core_step = 0
        
        #For the program to show the result faster, we will show results i.e. 
        #regions every 10 integration step.
        show_result = 10
        
        #Our core limit, as given on the paper, is 0.995 L_sun.
        #For that, we define the following.
        core_limit = 0.995
        
        #We define the boundaries of the R values on both x and y axes so that
        #it is between -1 to 1. (We kind of define a coordinate system)
        #Note that we chose 1.25 so that the Cross-Section of the Star doesn't
        #fill all of the plot!
        #Note that r_values[0] should be 1 so we need to Normalize the Radius
        #values before putting it onto the function!
        rmax = 1.25 * r_values[0]
        
        #We set the x and y axis limits to be between -1 to 1.
        #Note that it actually isn't -1 to 1 since we multiplied with 1.25
        #previously.
        ax.set_xlim(-rmax, rmax)
        ax.set_ylim(-rmax, rmax)
        
        #We also set the aspect of the axes to be equal so that our plot is
        #somewhat "circular".
        ax.set_aspect("equal")
        
        #We also set a variable to keep track of the integration steps and
        #figure when to show the results.
        x = show_result
        
        #Now we loop over the arrays.
        for i in range(0, n - 1):
            #We add 1 to our "show" variable.
            x += 1
            
            #If x is Larger or Equal to the show_result, we will show the
            #result.
            #Note that in that case, we will set x back to zero at the end of
            #the if statement we have defined!
            if x >= show_result:
                #If the Luminosity is larger than 0.995 limit, we are at the
                #outside core. => Shown with Red and Yellow on the Paper!
                if (L_values[i] > core_limit):
                    #In case of the Convective Flux being larger than 0, we
                    #call it the Outside Core with Convection!
                    if (FC_values[i] > 0.0):
                        #We color the Radius Values with FC > 0 and L > 0.995 
                        #as red!
                        circle_red = plt.Circle((0, 0), r_values[i], color = "red", fill = False)
                        
                        #We add the Red Circle we have created to the axes!
                        ax.add_artist(circle_red)
                        
                        #We update last_conv_step until the Convective Flux is 
                        #zero i.e. the last time this is updated is the end of 
                        #Convection Region Outside Core!
                        last_conv_step = i
                    
                    #In the case of the Convective Flux being equal to zero we
                    #define that region Outside Core with Radiation.
                    else:
                        #We color the Radius Values with FC = 0 and L > 0.995
                        #as Yellow!
                        circle_yellow = plt.Circle((0, 0), r_values[i], color = "yellow", fill = False)
                        
                        #We add the Yellow Circle we have created to the axes!
                        ax.add_artist(circle_yellow)
                
                #If Luminosity is smaller than 0.995 limit, we are at the
                #inside core. => Shown with Cyan and Blue on the paper!
                else:
                    #We update first_core_step at the start of the loop because
                    #we are now at the Core and we want to know how much of R0
                    #is the core.
                    if(first_core_step == 0):
                        first_core_step = i
                    
                    #In the case of Convective Flux being larger than 0, we
                    #call it the Inside Core with Convection!
                    if (FC_values[i] > 0.0):
                        #We color the Radius Values with FC > 0 and L < 0.995 
                        #as blue!
                        circle_blue = plt.Circle((0, 0), r_values[i], color = "blue", fill = False)
                        
                        #We add the Blue Circle we have created to the axes!
                        ax.add_artist(circle_blue)
                        
                    #In the case of the Convective Flux being equal to zero we
                    #define that region Inside Core with Radiation.
                    else:
                        #We color the Radius Values with FC = 0 and L < 0.995 
                        #as cyan!
                        circle_cyan = plt.Circle((0, 0), r_values[i], color='cyan', fill = False)
                        
                        #We add the Cyan Circle we have created to the axes!
                        ax.add_artist(circle_cyan)
                
                #We set x zero so that we can start the count over again until
                #it reaches 10 and we plot again.
                x = 0
        
        
        #We plot Circles again but this time out of the range of the plot.
        #We do this to make a legend that shows which color represents which
        #region.
        #Note that this time fill = True.
        circle_red = plt.Circle((2 * rmax, 2 * rmax), 0.1 * rmax, color = "red", fill = True)
        circle_yellow = plt.Circle((2 * rmax, 2 * rmax), 0.1 * rmax, color = "yellow", fill = True)
        circle_cyan = plt.Circle((2 * rmax, 2 * rmax), 0.1 * rmax, color = "cyan", fill = True)
        circle_blue = plt.Circle((2 * rmax, 2 * rmax), 0.1 * rmax, color = "blue", fill = True)
        
        #We create our legend using the extra circles we have plotted.
        ax.legend([circle_red, circle_yellow, circle_cyan, circle_blue],
                  ["Convection Outside Core", "Radiation Outside Core", 
                   "Radiation Inside Core", "Convection Inside Core"])
        
        #We add labels.
        plt.xlabel("$R$")
        plt.ylabel("$R$")
        plt.title(f"Cross-Section of Star Model \n (Outer Convection Layer: {(r_values[0] - r_values[last_conv_step])/r_values[0] * 100:.1f}% of $R_0$) & (The Core: {(r_values[first_core_step])/r_values[0] * 100:.1f}% of $R_0$)")
        plt.tight_layout()

        #We show the plot.
        plt.show()

############################# FINAL TEST ######################################

#Now we use the Class and its methods.
#For this we initilize it and call ODE_solver and plot functions.
if __name__ == "__main__":
    #We define our Object from the Class.
    star = Star_model()
    
    ########################## OPACITY SANITY CHECK ###########################
    
    #We first do Sanity Check for Opacity Interpolation.
    log_R_given = [-6.00, -5.95, -5.80, -5.70, -5.55, -5.95, -5.95, -5.95,
                   -5.80, -5.75, -5.70, -5.55, -5.50]
    
    log_T_given = [3.750, 3.755, 3.755, 3.755, 3.755, 3.770, 3.780, 3.795, 
                   3.770, 3.775, 3.780, 3.795, 3.800]
    
    expected_kappa = [2.84, 3.11, 2.68, 2.46, 2.12, 4.70, 6.25, 9.45, 4.05, 
                      4.43, 4.94, 6.89, 7.69]
    
    star.get_opacity(0, 0, log_R_given, log_T_given, expected_kappa)
    
    ###########################################################################
    
    ################### TOTAL GENERATED ENERGY SANITY CHECK ###################
    
    #We first do Sanity Check for Opacity Interpolation.
    log_R_given = [-6.00, -5.95]
    
    log_T_given = [3.750, 3.755]
    
    expected_epsilon = [1.012, 2.415]
    
    star.get_total_energy(0, 0, log_R_given, log_T_given, expected_epsilon)
    
    ###########################################################################
    
    #We solve the Stellar Structure Equations using ODE_solver from our object.
    r, P, L, T, rho, epsilon, m, nabla, nabla_star, nabla_stable, nabla_ad, FC, FR, F = star.ODE_solver()
    
    #We plot the Parameters using plot_parameters from our object.
    star.plot_parameters(m, r, rho, P, T, L, nabla_star, nabla_stable, nabla_ad)
    
    #We plot the Cross-Section of Star by using plot_cross_section from our
    #object.
    #For flux, we generate a list so that if F[i] is non-zero we get the ratio
    #and else if F[i] is zero we put zero.
    #We do this to AVOID DIVISION WITH ZERO WHEN F[i] = 0!
    star.plot_cross_section(r[ : star.end_step] / star.R_sun, 
                            L[ : star.end_step] / star.L_sun, 
                            [FC[i]/F[i] if F[i] else 0 for i in range(len(FC[:star.end_step]))])




