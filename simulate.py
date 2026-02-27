"""
Chemical ODE simulation for reversible two- and three-species systems.

Two-species:   A <-> C
Three-species: A <-> B <-> C
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pints

class FeFeSimulator:
    def __init__(self, E_filename, blank_filename):
        with open(E_filename, "r") as f:
            edata = np.loadtxt(f, skiprows=1, delimiter=",")
        self.times     = edata[:, 0]
        self.potential = edata[:, 1]
        with open(blank_filename, "r") as f:
            bdata = np.loadtxt(f, skiprows=1, delimiter=",")
        self.cap_current=bdata[:,1]
        self._interp   = interp1d(
            self.times, self.potential,
            kind='previous',
            bounds_error=False,
        )
        self.params_set=False
        self.boundaries_set=False
    def lookup(self, t):
        return float(self._interp(t))

    def _two_species_odes(self, t, y, params):
        """dA/dt, dC/dt for A <-> C."""
        A, C = y
        k_inact     = params["k_inact"]
        k_react     = params["k_react"]
        k_react_exp = params["k_react_exp"]
        k_deg       = params["k_deg"]
        pot = np.exp(k_react_exp * self.lookup(t))
        dA = -k_inact * A + k_react * pot * C - k_deg * A
        dC =  k_inact * A - k_react * pot * C - k_deg * C
        return [dA, dC]

    def _three_species_odes(self, t, y, params):
        """dA/dt, dB/dt, dC/dt for A <-> B <-> C."""
        A, B, C = y
        k_AB        = params["k_AB"]
        k_BA        = params["k_BA"]
        k_inact     = params["k_inact"]
        k_react     = params["k_react"]
        k_react_exp = params["k_react_exp"]
        k_deg       = params["k_deg"]
        pot = np.exp(k_react_exp * self.lookup(t))
        dA = -k_AB * A + k_BA * B - k_deg * A
        dB =  k_AB * A - k_BA * B - k_inact * B + k_react * pot * C - k_deg * B
        dC =  k_inact * B - k_react * pot * C - k_deg * C
        return [dA, dB, dC]

    # -----------------------------------------------------------------------
    # Optimisation helpers
    # -----------------------------------------------------------------------

    def set_param_names(self, param_names: list, fixed_params: dict={}):
        """
        Define the ordered list of parameters to optimise and any fixed
        parameters that will not be varied (e.g. initial conditions).

        Parameters
        ----------
        param_names : list of str
            Names of the parameters to optimise, in the order they will
            appear in the list passed to simulate().
        fixed_params : dict, optional
            Parameters held constant during optimisation.

        Raises
        ------
        ValueError
            If the union of param_names and fixed_params does not cover
            all parameters required by the chosen model.
        """
        self.param_names  = list(param_names)
        self.all_params=self.param_names+list(fixed_params.keys())
        in_both=set(param_names).intersection(set(fixed_params.keys()))
        if len(in_both)>0:
            raise ValueError("The following parameters are defined twice {0}".format(in_both))
        self.three_species = "init_1" in self.all_params 
        
        required_two   = {"k_inact", "k_react", "k_react_exp", "k_deg","current_conversion", "cap_scaling"}
        required_three = required_two | {"init_1","k_AB", "k_BA"}
        required = required_three if self.three_species else required_two

        all_defined = set(self.all_params) 
        missing = required - all_defined
        if missing:
            raise ValueError(f"Missing required parameter(s): {missing}")
        self.params_set=True
        self.fixed_parameters=fixed_params
    def set_boundaries(self,boundaries):
        """
        Set lower and upper bounds for each optimisation parameter.

        """
        self.boundaries=boundaries
        self.lower=np.array([boundaries[x][0] for x in self.param_names])
        self.upper=np.array([boundaries[x][1] for x in self.param_names])
        self.boundaries_set=True
    def unnormalise(self, param_list):
        """
        Map normalised values in [0, 1] to true parameter values.

        Parameters
        ----------
        x : array-like
            Normalised values, one per optimisation parameter.

        Returns
        -------
        np.ndarray
            True parameter values scaled to [lower, upper].
        """
        return self.lower + np.array(param_list) * (self.upper - self.lower)

    def list_to_dict(self, param_list) -> dict:
        """
        Wrap an (unnormalised) parameter list into a full parameter dict,
        merging with any fixed_params set via set_param_names().

        Parameters
        ----------
        param_list : array-like
            Unnormalised parameter values in the same order as param_names.

        Returns
        -------
        dict
            Complete parameter dictionary ready for the ODE solver.
        """
        params = dict(zip(self.param_names, param_list))
        return params
    def n_outputs(self):
        return 1
    def n_parameters(self):
        return len(self.param_names)
    # -----------------------------------------------------------------------
    def dimensional_simulate(self, param_dict: dict, times: np.ndarray):
        """
        Wrapper around simulate() for use with a dimensional (unnormalised)
        parameter dict, e.g. as returned by optimise().

        Extracts values in param_names order, normalises to [0, 1] using the
        stored boundaries, then delegates to simulate().
        """
        param_list = [param_dict[k] for k in self.param_names]
        normalised = [
            (v - self.boundaries[k][0]) / (self.boundaries[k][1] - self.boundaries[k][0])
            for k, v in zip(self.param_names, param_list)
        ]
        return self.simulate(normalised, times)

    def simulate(self, param_list: list, times: np.ndarray) -> dict:
        """
        Simulate a reversible two- or three-species chemical ODE system.

        The system is detected automatically from the keys present in `parameters`:
        - Two-species   (A <-> C)       if 'B0' is absent
        - Three-species (A <-> B <-> C) if 'B0' is present

        Parameters
        ----------
        parameters : dict
            Rate constants and initial concentrations.

            Two-species keys:
                A0, C0               -- initial concentrations
                k_inact, k_react     -- A <-> C rate constants
                k_react_exp          -- exponential scaling of potential
                k_deg                -- degradation rate

            Three-species keys (all of the above, plus):
                B0                   -- initial concentration of B
                k_AB, k_BA           -- A <-> B rate constants

        times : np.ndarray
            Monotonically increasing time points at which to return the solution.

        Returns
        -------
        dict
            Keys are species names ('A', 'C', and optionally 'B'); values are
            1-D np.ndarrays of concentrations aligned with `times`.

        Raises
        ------
        ValueError
            If required parameter keys are missing.
        """
        if self.params_set==False:
            raise ValueError("Please set parameter names via `set_param_names()`")
        if self.boundaries_set==False:
            raise ValueError("Please set parameter names via `set_boundaries()`")
        if any([x<0 for x in param_list]) or any([x>1 for x in param_list]):
            raise ValueError("simulate() only works with normalised parameters. Please try `dimensional_simulate()`")
        parameters=self.unnormalise(param_list)
        parameters=self.list_to_dict(parameters)
        parameters={**parameters, **self.fixed_parameters}
        t_span = (float(times[0]), float(times[-1]))
        kwargs = dict(t_eval=times, method="RK45", rtol=1e-8, atol=1e-10)
        
        if not self.three_species:
            A0 = 1
            C0 = 0
            sol = solve_ivp(self._two_species_odes, t_span, [A0, C0],
                            args=(parameters,), **kwargs)
            self.quantities={"A":sol.y[0], "C":sol.y[1]}
            current= sol.y[0]*parameters["current_conversion"]
        else:
            A0 = parameters["init_1"]
            B0 = 1-parameters["init_1"]
            C0 = 0
            sol = solve_ivp(self._three_species_odes, t_span, [A0, B0, C0],
                        args=(parameters,), **kwargs)
            self.quantities={"A":sol.y[0], "C":sol.y[2], "B":sol.y[1]}
            current= sol.y[0]*parameters["current_conversion"] + sol.y[1]*parameters["current_conversion"]/2
        return current+self.cap_current[np.where((self.times>=t_span[0]) & (self.times<=t_span[1]))]*parameters["cap_scaling"]#
    
    def optimise(self, times, current_data, **kwargs):
        self.current_data=current_data
        if "repeats" not in kwargs:
            kwargs["repeats"]=10
        if "parallel" not in kwargs:
            kwargs["parallel"] =True
        if "threshold" not in kwargs:
            kwargs["threshold"]=1e-4
        best_score=1e23
        problem = pints.SingleOutputProblem(self, times, current_data)
        score = pints.SumOfSquaresError(problem)
        boundaries = pints.RectangularBoundaries(np.zeros(len(self.param_names)), np.ones(len(self.param_names)))    
        
        for i in range(0, kwargs["repeats"]):
            x0 = np.random.rand(len(self.param_names))
            sigma0 = [0.1 for x in range(0, len(self.param_names))]
            opt = pints.OptimisationController(
                score,
                x0,
                sigma0,
                boundaries,
                method=pints.CMAES
            )
            opt.set_parallel(kwargs["parallel"])
            opt.set_max_unchanged_iterations(iterations=200, threshold=kwargs["threshold"])
            found_parameters, found_value = opt.run()
            if found_value<best_score:
                best_score=found_value
                best_params=found_parameters
                
                unnormed=self.unnormalise(best_params)
                print(list(unnormed))
        return dict(zip(self.param_names, unnormed))

        



