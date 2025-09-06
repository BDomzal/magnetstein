import numpy as np
from time import time
from masserstein import Spectrum
from masserstein import NMRSpectrum
import pulp as lp
from warnings import warn
import tempfile
from tqdm import tqdm
from pulp.apis import LpSolverDefault
from masserstein import misc
import math
from copy import deepcopy


def intensity_generator(confs, mzaxis):
        """
        Generates intensities from spectrum represented as a confs list,
        over ppm (or m/z) values from mzaxis.
        Assumes mzaxis and confs are sorted and returns consecutive intensities.
        """
        mzaxis_id = 0
        mzaxis_len = len(mzaxis)
        for mz, intsy in confs:
            while mzaxis[mzaxis_id] < mz:
                yield 0.
                mzaxis_id += 1
                if mzaxis_id == mzaxis_len:
                    return
            if mzaxis[mzaxis_id] == mz:
                yield intsy
                mzaxis_id += 1
                if mzaxis_id == mzaxis_len:
                    return
        for i in range(mzaxis_id, mzaxis_len):
                yield 0.


def dualdeconv2(exp_sp, thr_sps, penalty, quiet=True, solver=LpSolverDefault,
                warm_start_values=None):
        """
        This function solves linear program describing optimal transport of signal between the mixture's spectrum
        and the list of components' spectra. Additionally, an auxiliary point is introduced in order to
        remove noise from the mixture's spectrum, as described by Ciach et al., 2020. 
        _____
        Parameters:
            exp_sp: Spectrum object
                Mixture's spectrum.
            thr_sp: list of Spectrum objects
                List of components (i.e. reference, query, theoretical) spectra.
            penalty: float
                Denoising penalty.
            solver: 
                Which solver should be used. In case of problems with the default solver,
                lp.GUROBI() is recommended (note that it requires obtaining a licence).
                To see all solvers available at your machine execute: pulp.listSolvers(onlyAvailable=True) or lp.listSolvers(onlyAvailable=True).
            warm_start_values:
                List of tuples with variable as the first element and value of the variable as the second element.
                Use this argument, if you want the solver to be warm-started, i.e. to get some initial values of variables to start solving from.
                You can extract these values from the previous run of the function using 'output_warm_start_values' key.
                Important note: if you want to use this argument, you need to set warm_start=True for the used solver.
                For example: lp.GUROBI(warm_start=True).
        _____
        Returns: dict
            Dictionary with the following entries:
            - probs: List containing proportions of consecutive components' spectra in the mixture's
            spectrum. Note that they do not have to sum up to 1, because some part of the signal can be noise.
            - trash: Amount of noise in the consecutive ppm (or m/z) points from common horizontal axis.
            - fun: Optimal value of the objective function.
            - status: Status of the linear program.
            - output_warm_start_values: List of tuples with variable as the first element and optimal value of the variable as the second element.
                These values can be reused in the next run of the function, by setting argument warm_start_values to output_warm_start_values 
                obtained in the current run.
        """
        start = time()
        exp_confs = exp_sp.confs.copy()
        thr_confs = [thr_sp.confs.copy() for thr_sp in thr_sps]

        # Normalization check:
        assert np.isclose(sum(x[1] for x in exp_confs) , 1), "Mixture's spectrum not normalized."
        for i, thrcnf in enumerate(thr_confs):
                assert np.isclose(sum(x[1] for x in thrcnf), 1), "Component's spectrum %i is not normalized." % i

        # Computing a common horizontal axis for all spectra
        exp_confs = [(m, i) for m, i in exp_confs]
        thr_confs = [[(m, i) for m, i in cfs] for cfs in thr_confs]
        common_horizontal_axis = set(x[0] for x in exp_confs)
        common_horizontal_axis.update(x[0] for s in thr_confs for x in s)
        common_horizontal_axis = sorted(common_horizontal_axis)
        if not quiet:
                print("Common horizontal axis computed")
        n = len(common_horizontal_axis)
        k = len(thr_confs)

        # Computing lengths of intervals between ppm (or mz) measurements (l_i variables)
        interval_lengths = [common_horizontal_axis[i+1] - common_horizontal_axis[i] for i in range(n-1)]
        if not quiet:
                print("Interval lengths computed")

        # linear program:
        program = lp.LpProblem('Dual L1 regression sparse', lp.LpMaximize)
        if not quiet:
                print("Linear program initialized")
        # variables:
        lpVars = []
        for i in range(n):
                lpVars.append(lp.LpVariable('Z%i' % (i+1), None, penalty, lp.LpContinuous))
        ##        # in case one would like to explicitly forbid non-experimental abyss:
        ##        if V[i] > 0:
        ##            lpVars.append(lp.LpVariable('W%i' % (i+1), None, penalty, lp.LpContinuous))
        ##        else:
        ##            lpVars.append(lp.LpVariable('W%i' % (i+1), None, None, lp.LpContinuous))
        if not quiet:
                print("Variables created")
        # objective function:
        exp_vec = intensity_generator(exp_confs, common_horizontal_axis)  # generator of mixture's intensity observations
        program += lp.lpSum(v*x for v, x in zip(exp_vec, lpVars)), 'Dual_objective'
        # constraints:
        for j in range(k):
                thr_vec = intensity_generator(thr_confs[j], common_horizontal_axis)
                program += lp.lpSum(v*x for v, x in zip(thr_vec, lpVars) if v > 0.) <= 0, 'P%i' % (j+1)
        if not quiet:
                print('tsk tsk')
        ##    for i in range(n-1):
        ##        program += lpVars[i]-lpVars[i+1] <= interval_lengths[i], 'EpsPlus %i' % (i+1)
        ##        program += lpVars[i] - lpVars[i+1] >=  -interval_lengths[i], 'EpsMinus %i' % (i+1)
        for i in range(n-1):
                program +=  lpVars[i] - lpVars[i+1]  <=  interval_lengths[i], 'EpsPlus_%i' % (i+1)
                program +=  lpVars[i] - lpVars[i+1]  >= -interval_lengths[i], 'EpsMinus_%i' % (i+1)
        if not quiet:
                print("Constraints written")
        #program.writeLP('WassersteinL1.lp')
        if not quiet:
                print("Starting solver")

        if warm_start_values is not None and len(warm_start_values) == len(program.variables()):
            warm_start_values = dict(warm_start_values)
            for var in program.variables():
                try:
                    var.setInitialValue(warm_start_values[str(var)])
                except ValueError:
                    pass
        else:
            if not quiet:
                print('Current mixture spectrum (' + '+str(i)' + \
                        ') has different chemical shift axis than the previous one.\
                        Therefore, estimation for this spectrum will be performed \
                        without using information from the previous time point.')

        LpSolverDefault.msg = not quiet
        program.solve(solver = solver)
        end = time()
        if not quiet:
                print("Solver finished.")
                print("Status:", lp.LpStatus[program.status])
                print("Optimal value:", lp.value(program.objective))
                print("Time:", end - start)
        constraints = program.constraints
        probs = [round(constraints['P%i' % i].pi, 12) for i in range(1, k+1)]
        exp_vec = list(intensity_generator(exp_confs, common_horizontal_axis))
        # 'if' clause below is to restrict returned abyss to mixture's confs
        abyss = [round(x.dj, 12) for i, x in enumerate(lpVars)]
        # note: accounting for number of summands in checking of result correctness,
        # because summation of many small numbers introduces numerical errors
        if not np.isclose(sum(probs)+sum(abyss), 1., atol=len(abyss)*1e-03):
                warn("""In dualdeconv2:
                Proportions of signal and noise sum to %f instead of 1.
                This may indicate improper results.
                Please check the deconvolution results and consider reporting this warning to the authors.
                                    """ % (sum(probs)+sum(abyss)))

        return {"probs": probs, "trash": abyss, "fun": lp.value(program.objective), "status": program.status,
                "common_horizontal_axis": common_horizontal_axis,
                "output_warm_start_values": [(v, v.varValue) for v in program.variables()]}


def dualdeconv2_alternative(exp_sp, thr_sps, penalty, quiet=True, solver=LpSolverDefault,
                                warm_start_values=None):

        """
        Alternative version of dualdeconv2 - using .pi instead of .dj to extract optimal values of variables.
        Slower, but .pi is better documented than .dj in pulp. Gives the same results as dualdeconv2.
        This function solves linear program describing optimal transport of signal between the mixture's spectrum
        and the list of components' spectra. Additionally, an auxiliary point is introduced in order to
        remove noise from the mixture's spectrum, as described by Ciach et al., 2020. 
        _____
        Parameters:
            exp_sp: Spectrum object
                Mixture's spectrum.
            thr_sp: list of Spectrum objects
                List of components' (i.e. reference, query, theoretical) spectra.
            penalty: float
                Denoising penalty.
            solver: 
                Which solver should be used. In case of problems with the default solver,
                lp.GUROBI() is recommended (note that it requires obtaining a licence).
                To see all solvers available at your machine execute: pulp.listSolvers(onlyAvailable=True) or lp.listSolvers(onlyAvailable=True).
            warm_start_values:
                List of tuples with variable as the first element and value of the variable as the second element.
                Use this argument, if you want the solver to be warm-started, i.e. to get some initial values of variables to start solving from.
                You can extract these values from the previous run of the function using 'output_warm_start_values' key.
                Important note: if you want to use this argument, you need to set warm_start=True for the used solver.
                For example: lp.GUROBI(warm_start=True).
        _____
        Returns: dict
            Dictionary with the following entries:
            - probs: List containing proportions of consecutive components' spectra in the mixture's
            spectrum. Note that they do not have to sum up to 1, because some part of the signal can be noise.
            - trash: Amount of noise in the consecutive ppm (or m/z) points from common horizontal axis.
            - fun: Optimal value of the objective function.
            - status: Status of the linear program.
            - output_warm_start_values: List of tuples with variable as the first element and optimal value of the variable as the second element.
                These values can be reused in the next run of the function, by setting argument warm_start_values to output_warm_start_values 
                obtained in the current run.
        """

        start = time()
        exp_confs = exp_sp.confs.copy()
        thr_confs = [thr_sp.confs.copy() for thr_sp in thr_sps]

        # Normalization check:
        assert np.isclose(sum(x[1] for x in exp_confs) , 1), "Mixture's spectrum not normalized."
        for i, thrcnf in enumerate(thr_confs):
                assert np.isclose(sum(x[1] for x in thrcnf), 1), "Component's spectrum %i not normalized." % i

        # Computing a common horizontal axis for all spectra
        exp_confs = [(m, i) for m, i in exp_confs]
        thr_confs = [[(m, i) for m, i in cfs] for cfs in thr_confs]
        common_horizontal_axis = set(x[0] for x in exp_confs)
        common_horizontal_axis.update(x[0] for s in thr_confs for x in s)
        common_horizontal_axis = sorted(common_horizontal_axis)
        if not quiet:
                print("Common horizontal axis computed")
        n = len(common_horizontal_axis)
        k = len(thr_confs)

        # Computing lengths of intervals between ppm (or mz) measurements (l_i variables)
        interval_lengths = [common_horizontal_axis[i+1] - common_horizontal_axis[i] for i in range(n-1)]
        if not quiet:
                print("Interval lengths computed")

        # linear program:
        program = lp.LpProblem('Dual L1 regression sparse', lp.LpMaximize)
        if not quiet:
                print("Linear program initialized")
        # variables:
        lpVars = []
        for i in range(n):
                lpVars.append(lp.LpVariable('Z%i' % (i+1), None, None, lp.LpContinuous))
        ##        # in case one would like to explicitly forbid non-experimental abyss:
        ##        if V[i] > 0:
        ##            lpVars.append(lp.LpVariable('W%i' % (i+1), None, penalty, lp.LpContinuous))
        ##        else:
        ##            lpVars.append(lp.LpVariable('W%i' % (i+1), None, None, lp.LpContinuous))
        if not quiet:
                print("Variables created")
        # objective function:
        exp_vec = intensity_generator(exp_confs, common_horizontal_axis)  # generator of mixture's intensity observations
        program += lp.lpSum(v*x for v, x in zip(exp_vec, lpVars)), 'Dual_objective'
        # constraints:
        for j in range(k):
                thr_vec = intensity_generator(thr_confs[j], common_horizontal_axis)
                program += lp.lpSum(v*x for v, x in zip(thr_vec, lpVars) if v > 0.) <= 0, 'P%i' % (j+1)
        if not quiet:
                print('tsk tsk')
        for i in range(n):
                program += lpVars[i] <= penalty, 'g%i' % (i+1)
        for i in range(n-1):
                program +=  lpVars[i] - lpVars[i+1]  <=  interval_lengths[i], 'EpsPlus_%i' % (i+1)
                program +=  lpVars[i] - lpVars[i+1]  >= -interval_lengths[i], 'EpsMinus_%i' % (i+1)
        if not quiet:
                print("Constraints written")
        #program.writeLP('WassersteinL1.lp')
        if not quiet:
                print("Starting solver")

        if warm_start_values is not None and len(warm_start_values) == len(program.variables()):
            warm_start_values = dict(warm_start_values)
            for var in program.variables():
                try:
                    var.setInitialValue(warm_start_values[str(var)])
                except ValueError:
                    pass
        else:
            if not quiet:
                print('Current mixture spectrum (' + '+str(i)' + \
                        ') has different chemical shift axis than the previous one.\
                        Therefore, estimation for this spectrum will be performed \
                        without using information from the previous time point.')

        LpSolverDefault.msg = not quiet
        program.solve(solver = solver)
        end = time()
        if not quiet:
                print("Solver finished.")
                print("Status:", lp.LpStatus[program.status])
                print("Optimal value:", lp.value(program.objective))
                print("Time:", end - start)
        constraints = program.constraints
        probs = [round(constraints['P%i' % i].pi, 12) for i in range(1, k+1)]
        exp_vec = list(intensity_generator(exp_confs, common_horizontal_axis))
        abyss = [round(constraints['g%i' % i].pi, 12) for i in range(1, n+1)]
        # note: accounting for number of summands in checking of result correctness,
        # because summation of many small numbers introduces numerical errors
        if not np.isclose(sum(probs)+sum(abyss), 1., atol=len(abyss)*1e-03):
                warn("""In dualdeconv2_alternative:
                Proportions of signal and noise sum to %f instead of 1.
                This may indicate improper results.
                Please check the deconvolution results and consider reporting this warning to the authors.
                                    """ % (sum(probs)+sum(abyss)))

        return {"probs": probs, "trash": abyss, "fun": lp.value(program.objective), "status": program.status,
                "common_horizontal_axis": common_horizontal_axis,
                "output_warm_start_values": [(v, v.varValue) for v in program.variables()]}



def dualdeconv3(exp_sp, thr_sps, penalty, penalty_th, quiet=True, solver=LpSolverDefault,
                warm_start_values=None):

        """
        This function solves linear program describing optimal transport of signal between 
        the mixture's spectrum and the list of components' spectra. 
        Two auxiliary points are introduced in order to remove noise from the mixture's spectrum
        and from the combination of components' spectra. 
        Transport of signal between the two auxiliary points is explicitly forbidden.
        Mathematically, this formulation is equivalent to the one implemented in dualdeconv4
        and both give the same results up to roundoff errors.
        _____
        Parameters:
            exp_sp: Spectrum object
                Mixture's spectrum.
            thr_sp: Spectrum object
                List of components' (i.e. reference, query, theoretical) spectra.
            penalty: float
                Denoising penalty for the mixture's spectrum.
            penalty_th: float
                Denoising penalty for the components' spectra.
            solver: 
                Which solver should be used. In case of problems with the default solver,
                lp.GUROBI() is recommended (note that it requires obtaining a licence).
                To see all solvers available at your machine execute: pulp.listSolvers(onlyAvailable=True) or lp.listSolvers(onlyAvailable=True).
            warm_start_values:
                List of tuples with variable as the first element and value of the variable as the second element.
                Use this argument, if you want the solver to be warm-started, i.e. to get some initial values of variables to start solving from.
                You can extract these values from the previous run of the function using 'output_warm_start_values' key.
                Important note: if you want to use this argument, you need to set warm_start=True for the used solver.
                For example: lp.GUROBI(warm_start=True).
        _____
        Returns: dict
            Dictionary with the following entries:
            - probs: List containing proportions of consecutive components' spectra in the mixture's
            spectrum. Note that they do not have to sum up to 1, because some part of the signal can be noise.
            - noise_in_components: Proportion of noise present in the combination of components' spectra.
            - trash: Amount of noise in the consecutive ppm (or m/z) points from common horizontal axis.
            - components_trash: Amount of noise present in the combination of components'
            spectra in consecutive ppm (or m/z) points from common horizontal axis.
            - fun: Optimal value of the objective function.
            - status: Status of the linear program.
            - common horizontal axis: All the ppm (or m/z) values from the mixture's spectrum and from the components' 
            spectra in a sorted list. 
            - output_warm_start_values: List of tuples with variable as the first element and optimal value of the variable as the second element.
                These values can be reused in the next run of the function, by setting argument warm_start_values to output_warm_start_values 
                obtained in the current run.
        """

        start = time()
        exp_confs = exp_sp.confs.copy()
        thr_confs = [thr_sp.confs.copy() for thr_sp in thr_sps]

        # Normalization check:
        assert np.isclose(sum(x[1] for x in exp_confs) , 1), "Mixture's spectrum not normalized."
        for i, thrcnf in enumerate(thr_confs):
                assert np.isclose(sum(x[1] for x in thrcnf), 1), "Component's spectrum %i not normalized." % i

        # Computing a common horizontal axis for all spectra
        exp_confs = [(m, i) for m, i in exp_confs]
        thr_confs = [[(m, i) for m, i in cfs] for cfs in thr_confs]
        common_horizontal_axis = set(x[0] for x in exp_confs)
        common_horizontal_axis.update(x[0] for s in thr_confs for x in s)
        common_horizontal_axis = sorted(common_horizontal_axis)
        if not quiet:
                print("Common horizontal axis computed")
        n = len(common_horizontal_axis)
        k = len(thr_confs)

        # Computing lengths of intervals between ppm (or mz) measurements (l_i variables)
        interval_lengths = [common_horizontal_axis[i+1] - common_horizontal_axis[i] for i in range(n-1)]
        if not quiet:
                print("Interval lengths computed")

        # linear program:
        program = lp.LpProblem('Dual L1 regression sparse', lp.LpMaximize)
        if not quiet:
                print("Linear program initialized")

        # variables:
        lpVars = []
        try:
                for i in range(n-2):
                        lpVars.append(lp.LpVariable('Z%i' % (i+1), None, None, lp.LpContinuous))
                lpVars.append(lp.LpVariable('Z%i' % (n-1), -interval_lengths[n-2], interval_lengths[n-2], lp.LpContinuous))
        except IndexError: #linear program makes no sense if n<=2 (n-number of points in mixture's spectrum)
                pass
        lpVars.append(lp.LpVariable('Z%i' % (n), None, None, lp.LpContinuous))
        lpVars.append(lp.LpVariable('Z%i' % (n+1), None, None, lp.LpContinuous))
        lpVars.append(lp.LpVariable('Z%i' % (n+2), 0, None, lp.LpContinuous))
        lpVars.append(lp.LpVariable('Z%i' % (n+3), 0, None, lp.LpContinuous))

        if not quiet:
                print("Variables created")

        # objective function:
        exp_vec = intensity_generator(exp_confs, common_horizontal_axis)  # generator of mixture's intensity observations
        program += lp.lpSum(v*x for v, x in zip(exp_vec, lpVars[:n-1]+[0])).addInPlace(
                                lp.lpSum(v*x for v, x in zip([-1, 0, 0, -1], lpVars[n-1:]))), 'Dual_objective'

        # constraints:
        for j in range(k):
                thr_vec = intensity_generator(thr_confs[j], common_horizontal_axis)
                program += lp.lpSum(v*x for v, x in zip(thr_vec, lpVars[:n-1]+[0]) if v > 0.).addInPlace(
                                lp.lpSum(v*x for v, x in zip([-1, 0, 1, -1], lpVars[n-1:]))) <= 0, 'P_%i' % (j+1)

        exp_vec = intensity_generator(exp_confs, common_horizontal_axis)
        program += lp.lpSum(v*x for v, x in zip(exp_vec, lpVars[:n-1]+[0])).addInPlace(
                                lp.lpSum(v*x for v, x in zip([0, 1, -1, 0], lpVars[n-1:]))) <= 0, 'p0_prime'

        if not quiet:
                print('tsk tsk')

        for i in range(n-1):
                program +=  lpVars[i] - lpVars[n-1]  <=  penalty, 'g_%i' % (i+1)
                program +=  -lpVars[n] - lpVars[i] <= penalty_th, 'g_prime_%i' % (i+1)
        try:
                for i in range(n-2):
                        program += lpVars[i] - lpVars[i+1] <= interval_lengths[i], 'epsilon_plus_%i' % (i+1)
                        program += lpVars[i+1] - lpVars[i] <= interval_lengths[i], 'epsilon_minus_%i' % (i+1)
        except IndexError: #linear program makes no sense if n<=2 (n-number of points in mixture's spectrum)
                pass

        program += -lpVars[n-1] <= penalty, 'g_%i' % (n)
        program += -lpVars[n] <= penalty_th, 'g_prime_%i' % (n)

        if not quiet:
                print("Constraints written")

        if not quiet:
                print("Starting solver")

        if warm_start_values is not None and len(warm_start_values) == len(program.variables()):
            warm_start_values = dict(warm_start_values)
            for var in program.variables():
                try:
                    var.setInitialValue(warm_start_values[str(var)])
                except ValueError:
                    pass
        else:
            if not quiet:
                print('Current mixture spectrum (' + '+str(i)' + \
                        ') has different chemical shift axis than the previous one.\
                        Therefore, estimation for this spectrum will be performed \
                        without using information from the previous time point.')

        #Solving
        LpSolverDefault.msg = not quiet
        program.solve(solver = solver)
        end = time()
        if not quiet:
                print("Solver finished.")
                print("Status:", lp.LpStatus[program.status])
                print("Optimal value:", lp.value(program.objective))
                print("Time:", end - start)
        constraints = program.constraints
        probs = [round(constraints['P_%i' % i].pi, 12) for i in range(1, k+1)]
        p0_prime = round(constraints['p0_prime'].pi, 12)
        exp_vec = list(intensity_generator(exp_confs, common_horizontal_axis))
        abyss = [round(constraints['g_%i' % i].pi, 12) for i in range(1, n+1)]
        abyss_th = [round(constraints['g_prime_%i' % i].pi, 12) for i in range(1, n+1)]

        if not np.isclose(sum(probs)+sum(abyss), 1., atol=len(abyss)*1e-03):
                warn("""In dualdeconv3:
                Proportions of signal and noise sum to %f instead of 1.
                This may indicate improper results.
                Please check the deconvolution results and consider reporting this warning to the authors.
                                    """ % (sum(probs)+sum(abyss)))

        return {"probs": probs, "noise_in_components": p0_prime, "trash": abyss, "components_trash": abyss_th,
                "fun": lp.value(program.objective), "status": program.status, "common_horizontal_axis": common_horizontal_axis,
                "output_warm_start_values": [(v, v.varValue) for v in program.variables()]}


def dualdeconv4(exp_sp, thr_sps, penalty, penalty_th, quiet=True, solver=LpSolverDefault,
                warm_start_values=None):

        """
        This function solves linear program describing optimal transport of signal between the mixture's 
        spectrum and the list of components' spectra. 
        Two auxiliary points are introduced in order to remove noise from the mixture's spectrum
        and from the combination of components' spectra. 
        Transport of signal between the two auxiliary points is allowed (with cost equal to penalty + penalty_th),
        however, it is not optimal so it never occurs. Mathematically, this formulation is equivalent to the one 
        implemented in dualdeconv3 and both give the same results up to roundoff errors.
        _____
        Parameters:
            exp_sp: Spectrum object
                Mixture's spectrum.
            thr_sp: Spectrum object
                List of components' (i.e. reference, query, theoretical) spectra.
            penalty: float
                Denoising penalty for the mixture's spectrum.
            penalty_th: float
                Denoising penalty for the components spectra.
            solver: 
                Which solver should be used. In case of problems with the default solver,
                lp.GUROBI() is recommended (note that it requires obtaining a licence).
                To see all solvers available at your machine execute: pulp.listSolvers(onlyAvailable=True) or lp.listSolvers(onlyAvailable=True).
            warm_start_values:
                List of tuples with variable as the first element and value of the variable as the second element.
                Use this argument, if you want the solver to be warm-started, i.e. to get some initial values of variables to start solving from.
                You can extract these values from the previous run of the function using 'output_warm_start_values' key.
                Important note: if you want to use this argument, you need to set warm_start=True for the used solver.
                For example: lp.GUROBI(warm_start=True).
        
        _____
        Returns: dict
            Dictionary with the following entries:
            - probs: List containing proportions of consecutive components' spectra in the mixture's
            spectrum. Note that they do not have to sum up to 1, because some part of the signal can be noise.
            - noise_in_components: Proportion of noise present in the combination of components' spectra.
            - trash: Amount of noise in the consecutive ppm (or m/z) points from common horizontal axis.
            - components_trash: Amount of noise present in the combination of components'
            spectra in consecutive ppm (or m/z) points from common horizontal axis.
            - fun: Optimal value of the objective function.
            - status: Status of the linear program.
            - common horizontal axis: All the ppm (or m/z) values from the mixture's spectrum and from the components' spectra in a sorted list. 
            - output_warm_start_values: List of tuples with variable as the first element and optimal value of the variable as the second element.
                These values can be reused in the next run of the function, by setting argument warm_start_values to output_warm_start_values 
                obtained in the current run.
        """

        start = time()
        exp_confs = exp_sp.confs.copy()
        thr_confs = [thr_sp.confs.copy() for thr_sp in thr_sps]

        # Normalization check:
        assert np.isclose(sum(x[1] for x in exp_confs) , 1), "Mixture's spectrum not normalized."
        for i, thrcnf in enumerate(thr_confs):
                assert np.isclose(sum(x[1] for x in thrcnf), 1), "Component's spectrum %i not normalized." % i

        # Computing a common horizontal axis for all spectra
        exp_confs = [(m, i) for m, i in exp_confs]
        thr_confs = [[(m, i) for m, i in cfs] for cfs in thr_confs]
        common_horizontal_axis = set(x[0] for x in exp_confs)
        common_horizontal_axis.update(x[0] for s in thr_confs for x in s)
        common_horizontal_axis = sorted(common_horizontal_axis)
        if not quiet:
                print("Common horizontal axis computed")
        n = len(common_horizontal_axis)
        k = len(thr_confs)

        # Computing lengths of intervals between ppm(or mz) measurements (l_i variables)
        interval_lengths = [common_horizontal_axis[i+1] - common_horizontal_axis[i] for i in range(n-1)]
        if not quiet:
                print("Interval lengths computed")

        # linear program:
        program = lp.LpProblem('Dual L1 regression sparse', lp.LpMaximize)
        if not quiet:
                print("Linear program initialized")

        # variables:
        lpVars = []
        try:
                for i in range(n-2):
                        lpVars.append(lp.LpVariable('Z%i' % (i+1), None, None, lp.LpContinuous))
                lpVars.append(lp.LpVariable('Z%i' % (n-1), -interval_lengths[n-2], interval_lengths[n-2], lp.LpContinuous))
        except IndexError: #linear program makes no sense if n<=2 (n-number of points in mixture's spectrum)
                pass
        lpVars.append(lp.LpVariable('Z%i' % n, None, None, lp.LpContinuous))
        lpVars.append(lp.LpVariable('Z%i' % (n+1), None, None, lp.LpContinuous))
        lpVars.append(lp.LpVariable('Z%i' % (n+2), 0, None, lp.LpContinuous))
        lpVars.append(lp.LpVariable('Z%i' % (n+3), 0, None, lp.LpContinuous))
        if not quiet:
                print("Variables created")

        # objective function:
        exp_vec = intensity_generator(exp_confs, common_horizontal_axis)  # generator of mixture's intensity observations
        program += lp.lpSum(v*x for v, x in zip(exp_vec, lpVars[:n-1]+[0])).addInPlace(
                            lp.lpSum(v*x for v, x in zip([-1, 0, 0, -1], lpVars[n-1:]))), 'Dual_objective'

        # constraints:
        for j in range(k):
                thr_vec = intensity_generator(thr_confs[j], common_horizontal_axis)
                program += lp.lpSum(v*x for v, x in zip(thr_vec, lpVars[:n-1]+[0]) if v > 0.).addInPlace(
                            lp.lpSum(v*x for v, x in zip([-1, 0, 1, -1], lpVars[n-1:]))) <= -penalty, 'P_%i' % (j+1)

        exp_vec = intensity_generator(exp_confs, common_horizontal_axis)
        program += lp.lpSum(v*x for v, x in zip(exp_vec, lpVars[:n-1]+[0])).addInPlace(
                            lp.lpSum(v*x for v, x in zip([0, 1, -1, 0], lpVars[n-1:]))) <= penalty_th, 'p0_prime'

        if not quiet:
                print('tsk tsk')
        
        for i in range(n-1):
                program +=  lpVars[i] - lpVars[n-1]  <=  0, 'g_%i' % (i+1)
                program +=  -lpVars[i] - lpVars[n]  <= 0, 'g_prime_%i' % (i+1)
        try:
                for i in range(n-2):
                        program += lpVars[i] - lpVars[i+1] <= interval_lengths[i], 'epsilon_plus_%i' % (i+1)
                        program += lpVars[i+1] - lpVars[i] <= interval_lengths[i], 'epsilon_minus_%i' % (i+1)
        except IndexError: #linear program makes no sense if n<=2 (n-number of points in mixture's spectrum)
                pass

        program += -lpVars[n-1] <= 0, 'g_%i' % (n)
        program += -lpVars[n] <= 0, 'g_prime_%i' % (n)

        if not quiet:
                print("Constraints written")

        if not quiet:
                print("Starting solver")

        if warm_start_values is not None and len(warm_start_values) == len(program.variables()):
            warm_start_values = dict(warm_start_values)
            for var in program.variables():
                try:
                    var.setInitialValue(warm_start_values[str(var)])
                except ValueError:
                    pass
        else:
            if not quiet:
                print('Current mixture spectrum (' + '+str(i)' + \
                        ') has different chemical shift axis than the previous one.\
                        Therefore, estimation for this spectrum will be performed \
                        without using information from the previous time point.')

        #Solving
        LpSolverDefault.msg = not quiet
        program.solve(solver = solver)
        end = time()
        if not quiet:
                print("Solver finished.")
                print("Status:", lp.LpStatus[program.status])
                print("Optimal value:", lp.value(program.objective))
                print("Time:", end - start)
        constraints = program.constraints
        probs = [round(constraints['P_%i' % i].pi, 12) for i in range(1, k+1)]
        p0_prime = round(constraints['p0_prime'].pi, 12)
        exp_vec = list(intensity_generator(exp_confs, common_horizontal_axis))
        abyss = [round(constraints['g_%i' % i].pi, 12) for i in range(1, n+1)]
        abyss_th = [round(constraints['g_prime_%i' % i].pi, 12) for i in range(1, n+1)]
        if not np.isclose(sum(probs)+sum(abyss), 1., atol=len(abyss)*1e-03):
                warn("""In dualdeconv4:
                Proportions of signal and noise sum to %f instead of 1.
                This may indicate improper results.
                Please check the deconvolution results and consider reporting this warning to the authors.
                                    """ % (sum(probs)+sum(abyss)))

        return {"probs": probs, "noise_in_components": p0_prime, "trash": abyss, "components_trash": abyss_th, 
                "fun": lp.value(program.objective)+penalty, "status": program.status, "common_horizontal_axis": common_horizontal_axis,
                'output_warm_start_values': [(str(v), v.varValue) for v in program.variables()]}


def estimate_proportions(spectrum, query, MTD=0.25, MDC=1e-8,
                        MMD=-1, max_reruns=3, verbose=False, 
                        progress=False, MTD_th=0.22, solver=lp.GUROBI(),
                        what_to_compare='concentration',
                        warm_start_values=None):
    """
    Returns estimated proportions of components from query in mixture's spectrum.
    Performs initial filtering of components' and mixture's spectra to speed up the computations.
    _____
    Parameters:
    spectrum: Spectrum object
        The mixture's spectrum.
    query: list of Spectrum objects
        A list of components' spectra (reference spectra).
    MTD: Maximum Transport Distance, float
        Signal from mixture's spectrum will be transported up to this distance when estimating
        components proportions. This parameter is interpreted as denoising penalty for mixture.
        To disable denoising, set this parameter to large value (for example 1000). Default is 0.25. 
    MDC: Minimum Detectable Current, float
        If the spectrum of a component encompasses less than
        this amount of the total signal, it is assumed that this component
        is absent in the spectrum. Default is 1e-8.
    MMD: Maximum Mode Distance, float
        If there is no mixture's peak within this distance from the
        highest peak of spectrum of a component,
        it is assumed that this component is absent in the spectrum.
        Setting this value to -1 disables filtering. Default is -1.
    max_reruns: int
        Due to numerical errors, some partial results may be inaccurate.
        If this is detected, then those results are recomputed for a maximal number of times
        given by this parameter. Default is 3.
    verbose: bool
        Print diagnostic messages? Default is False.
    progress: bool
        Whether to display progress bars during work. Default is False.
    MTD_th: Maximum Transport Distance for components' spectra, float
        If presence of noise in components' spectra is not expected, 
        then this parameter should be set to None. Otherwise, set its value to some positive real number.
        Signal from components' spectra will be transported up to this distance 
        when estimating components' proportions. This parameter is interpreted as denoising penalty for components. Default is 0.22.
    solver: 
        Which solver should be used. We recommend using lp.GUROBI() (note that it requires obtaining a licence).
        To see all solvers available at your machine execute: pulp.listSolvers(onlyAvailable=True) or lp.listSolvers(onlyAvailable=True). 
        If you prefer to use Magnetstein without Gurobi set this argument to solver=LpSolverDefault.
        Note that using Magnetstein without Gurobi can result in long computation time and, in some cases, incorrect results.
    what_to_compare:
        Should the resulting proportions correspond to concentrations or area under the curve? Default is
        'concentration'. Alternatively can be set to 'area'. This argument is used only for NMR spectra.
    warm_start_values:
        List of lists of tuples with variable as the first element of the tuple and value of the variable as the second element of the tuple.
        Each of the nested lists corresponds to a chunk in exp_conf_chunKs.
        Use this argument, if you want the solver to be warm-started, i.e. to get some initial values of variables to start solving from.
        You can extract the values from the previous run of the function using 'output_warm_start_values' key.
        Important note: if you want to use this argument, you need to set warm_start=True for the used solver.
        For example: lp.GUROBI(warm_start=True).
    _____
    Returns: dict
        A dictionary with the following entries:
        - proportions: List of proportions of components' spectra.
        - Wasserstein distance: Value of Wasserstein distance between mixture's spectrum and component's spectra added 
        in computed proportions (taking into consideration removed signal).
        If what_to_compare='area' and MTD_th parameter is not equal to None, then the dictionary contains also 
        the following entries:
        - noise: List of intensities that could not be explained by the supplied formulas. 
        The intensities correspond to the ppm (or m/z) values of the mixture's spectrum.
        - noise_in_components: List of intensities from components' spectra
        that do not correspond to any intensities in the mixture's spectrum and therefore were 
        identified as noise. The intensities correspond to the ppm (or m/z) values from common horizontal axis.
        - proportion_of_noise_in_components: Proportion of noise present in the combination of components'
        spectra.
        - common_horizontal_axis: All the ppm (or m/z) values from the mixture's spectrum and from the components' 
        spectra in a sorted list. 
        - output_warm_start_values: List of lists of tuples with variable as the first element of the tuple and optimal value of the variable 
        as the second element of the tuple.
        Each of the nested lists corresponds to a chunk in exp_conf_chunKs.
        These values can be reused in the next run of the function, by setting argument warm_start_values to output_warm_start_values 
        obtained in the current run.
    """

    def progr_bar(x, **kwargs):
        if progress:
            return tqdm(x, **kwargs)
        else:
            return x

    try:
        exp_confs = spectrum.confs
    except:
        print("Could not retrieve the confs list. Is the supplied spectrum an object of class Spectrum?")
        raise

    
    assert what_to_compare=='concentration' or what_to_compare=='area', 'Comparison of %s is not supported' %what_to_compare

    if what_to_compare=='concentration':
        for i, sp in enumerate(query):
                assert sp.protons is not None, "Component " + str(i) +" doesn't have the number of protons defined. Please define sp.protons attribute."

    is_NMR_spectrum = [isinstance(sp, NMRSpectrum) for sp in [spectrum] + query]
    assert all(is_NMR_spectrum) or not any(is_NMR_spectrum), 'Spectra provided are of mixed types. \
            Please assert that either all or none of the spectra are NMR spectra.'
    nmr = all(is_NMR_spectrum)

    if not nmr:
        assert all(x[0] >= 0. for x in exp_confs), 'Found peaks with negative masses!'

    if any(x[1] < 0 for x in exp_confs):
        print("The mixture's spectrum cannot contain negative intensities. ")
        print("Setting negative intensities to zero.")
        exp_confs = [(ppm, intsy if intsy >= 0 else 0.) for ppm, intsy in exp_confs]

    if not abs(sum(x[1] for x in exp_confs) - 1.) < 1e-08:
        print("The mixture's spectrum is not normalized.")
        print("Normalizing mixture's spectrum.")
        scaling_factor = 1.0/math.fsum(v[1] for v in exp_confs)
        exp_confs = [(v[0], v[1]*scaling_factor) for v in exp_confs]

    assert abs(sum(x[1] for x in exp_confs) - 1.) < 1e-08, "The mixture's spectrum is not normalized."
                           
    k = len(query)
    proportions = [0.]*k

    if MTD_th is None:
        MTD_max = MTD
    else:
        MTD_max = max(MTD, MTD_th)

    preprocessed_query = []
    for i, q in enumerate(query):

        q_confs = q.confs

        if any(x[1] < 0 for x in q_confs):
            print("Component's spectrum %i cannot contain negative intensities." %i)
            print("Setting negative intensities to zero.")
            q_confs = [(ppm, intsy if intsy >= 0 else 0.) for ppm, intsy in q_confs]

        if not abs(sum(x[1] for x in q_confs) - 1.) < 1e-08:
            print("Component's spectrum %i is not normalized." %i)
            print("Normalizing component's spectrum %i." %i)
            scaling_factor = 1.0/math.fsum(v[1] for v in q_confs)
            q_confs = [(v[0], v[1]*scaling_factor) for v in q_confs]

        assert abs(sum(x[1] for x in q_confs) - 1.) < 1e-08, "Component's spectrum is not normalized."

        if not nmr:
            assert all(x[0] >= 0 for x in q.confs), "Component's spectrum %i has negative masses!" %i
            preprocessed_query.append(Spectrum(confs=q_confs))
        else:
            preprocessed_query.append(NMRSpectrum(confs=q_confs, protons=query[i].protons))
        
    # Initial filtering of formulas
    envelope_bounds = []
    filtered = []
    for i in progr_bar(range(k), desc = "Initial filtering of formulas"):
        s = preprocessed_query[i]
        mode = s.get_modal_peak()[0]
        mn = s.confs[0][0]
        mx = s.confs[-1][0]
        matching_current = MDC==0. or sum(x[1] for x in misc.extract_range(
                                                                        exp_confs, mn - MTD_max, mx + MTD_max)) >= MDC
        matching_mode = MMD==-1 or abs(misc.closest(exp_confs, mode)[0] - mode) <= MMD

        if matching_mode and matching_current:
            envelope_bounds.append((mn, mx, i))
        else:
            envelope_bounds.append((-1, -1, i))
            filtered.append(i)

    envelope_bounds.sort(key=lambda x: x[0])  # sorting by lower bounds
    if verbose:
        print("Removed components' spectra due to no matching peaks in mixture's spectrum:", filtered)
        print('Envelope bounds:', envelope_bounds)

    # Computing chunks
    chunkIDs = [0]*k  # Grouping of components' spectra
    # Note: order of chunkIDs corresponds to order of query, not the envelope bounds
    # chunk_bounds = intervals matching chunks, accounting for signal transport
    # order of chunk_bounds corresponds to increasing chunk ID,
    # so that chunk_bounds[0] is the interval for chunk nr 0
    chunk_bounds = []
    current_chunk = 0
    first_present = 0
    while envelope_bounds[first_present][0] == -1 and first_present < k-1:
        _, _, sp_id = envelope_bounds[first_present]
        chunkIDs[sp_id] = -1
        first_present += 1
    prev_mn, prev_mx, prev_id = envelope_bounds[first_present]
    for i in progr_bar(range(first_present, k), desc = "Computing chunks"):
        mn, mx, sp_id = envelope_bounds[i]
        if mn - prev_mx > 2*MTD_max:
            current_chunk += 1
            chunk_bounds.append( (prev_mn-MTD_max, prev_mx+MTD_max) )
            prev_mn = mn  # get lower bound of new chunk
        prev_mx = mx  # update the lower bound of current chunk
        chunkIDs[sp_id] = current_chunk
    chunk_bounds.append( (prev_mn-MTD_max, prev_mx+MTD_max) )
    nb_of_chunks = len(chunk_bounds)
    if verbose:
        print('Number of chunks: %i' % nb_of_chunks)
        print("ChunkIDs:", chunkIDs)
        print("Chunk bounds:", chunk_bounds)

    # Splitting the mixture's spectrum into chunks
    exp_conf_chunks = []  # list of indices of mixture's confs matching chunks
    current_chunk = 0
    matching_confs = []  # mixture's confs matching current chunk
    exp_confs_outside_chunks = []
    cur_bound = chunk_bounds[current_chunk]
    for conf_id, cur_conf in progr_bar(enumerate(exp_confs), desc = "Splitting the mixture's spectrum into chunks"):
        while cur_bound[1] < cur_conf[0] and current_chunk < nb_of_chunks-1:
            exp_conf_chunks.append(matching_confs)
            matching_confs = []
            current_chunk += 1
            cur_bound = chunk_bounds[current_chunk]
        if cur_bound[0] <= cur_conf[0] <= cur_bound[1]:
            matching_confs.append(conf_id)
        else:
            #those exp_confs that are outside all the chunks are kept in special list
            #later they will be attached to the vortex
            exp_confs_outside_chunks.append(cur_conf)
    exp_conf_chunks.append(matching_confs)
    chunk_TICs = [sum(exp_confs[i][1] for i in chunk_list) for chunk_list in exp_conf_chunks]
    if verbose:
        print("Ion currents in chunks:", chunk_TICs)

    # Deconvolving chunks:
    p0_prime = 0
    vortex =[]
    vortex_th = []
    common_horizontal_axis = []
    objective_function = 0
    exp_confs_in_almost_empty_chunks = []
    output_warm_start_values = []

    warm_start_possible = warm_start_values is not None and len(warm_start_values) == len(exp_conf_chunks)

    for current_chunk_ID, conf_IDs in progr_bar(enumerate(exp_conf_chunks), desc="Deconvolving chunks",
                                                                            total=len(exp_conf_chunks)):
        if verbose:
            print("Deconvolving chunk %i" % current_chunk_ID)
        if chunk_TICs[current_chunk_ID] < 1e-16:
            # nothing to deconvolve, pushing remaining signal to vortex
            if verbose:
                print('Chunk %i is almost empty - skipping deconvolution' % current_chunk_ID)
            for i in conf_IDs:
                #confs from very small chunks will be kept in a special list
                #later this list will be attached to vortex
                exp_confs_in_almost_empty_chunks.append(exp_confs[i])
        else:
            chunkSp = Spectrum('', empty=True)
            # Note: conf_IDs are monotonic w.r.t. conf mass,
            # so constructing a spectrum will not change the order
            # of confs supplied in the list below:
            chunkSp.set_confs([exp_confs[i] for i in conf_IDs])
            chunkSp.normalize()
            components_spectra_IDs = [i for i, c in enumerate(chunkIDs) if c == current_chunk_ID]
            thrSp = [preprocessed_query[i] for i in components_spectra_IDs]

            rerun = 0
            success = False
            while not success:
                    rerun += 1
                    if rerun > max_reruns:
                            raise RuntimeError("Failed to deconvolve a fragment of the mixture's spectrum:\
                                                 (%f, %f)" % chunk_bounds[current_chunk_ID])
                    if MTD_th is None:
                        if warm_start_possible:
                            dec = dualdeconv2(chunkSp, thrSp, MTD, quiet=True, solver=solver, warm_start_values=warm_start_values[current_chunk_ID])

                        else:
                            dec = dualdeconv2(chunkSp, thrSp, MTD, quiet=True, solver=solver, warm_start_values=None)
                            
                    else:
                        if warm_start_possible:
                            dec = dualdeconv4(chunkSp, thrSp, MTD, MTD_th, quiet=True, solver=solver, warm_start_values=warm_start_values[current_chunk_ID])
                        else:
                            dec = dualdeconv4(chunkSp, thrSp, MTD, MTD_th, quiet=True, solver=solver, warm_start_values=None)

                    if dec['status'] == 1:
                            success=True
                    else:
                            warn('Rerunning computations for chunk %i due to status %s' % (current_chunk_ID, 
                                                                                        lp.LpStatus[dec['status']]))
            if verbose:
                    print('Chunk %i deconvolution status:', lp.LpStatus[dec['status']])
                    print("Signal proportion in mixture's spectrum:", sum(dec['probs']))
                    print("Noise proportion in mixture's spectrum:", sum(dec['trash']))
                    print('Total explanation:', sum(dec['probs'])+sum(dec['trash']))
                    if MTD_th is not None:
                        print("Noise proportion in combination of components' spectra:", dec["noise_in_components"])
            for i, p in enumerate(dec['probs']):
                original_thr_spectrum_ID = components_spectra_IDs[i]
                proportions[original_thr_spectrum_ID] = p*chunk_TICs[current_chunk_ID]

            rescaled_vortex = [element*chunk_TICs[current_chunk_ID] for element in dec['trash']]
            vortex = vortex + rescaled_vortex
            common_horizontal_axis = common_horizontal_axis + dec['common_horizontal_axis']
            
            if MTD_th is not None:
                p0_prime = p0_prime + dec["noise_in_components"]*chunk_TICs[current_chunk_ID]
                rescaled_vortex_th = [element*chunk_TICs[current_chunk_ID] for element in dec['components_trash']]
                vortex_th = vortex_th + rescaled_vortex_th

            output_warm_start_values.append(dec['output_warm_start_values'])
                
            objective_function = objective_function + dec['fun']

    assert len(common_horizontal_axis) == len(vortex)
    if MTD_th is not None:
        assert len(common_horizontal_axis) == len(vortex_th)

    #confs from outside common_horizontal_axis are gathered in one list
    exp_confs_from_outside_cha = exp_confs_outside_chunks + exp_confs_in_almost_empty_chunks
    
    
    #appending these confs to vortex
    vortex = list(zip(common_horizontal_axis, vortex)) + exp_confs_from_outside_cha
    vortex = sorted(vortex, key = lambda x: x[0])
    common_horizontal_axis_v = [el[0] for el in vortex]
    vortex = [el[1] for el in vortex]
    if MTD_th is not None:
        #since common_horizontal_axis will be updated, we need to add new confs to vortex_th as well
        #those elements will always have intensities equal to zero, because they are from outside chunks
        vortex_th = list(zip(common_horizontal_axis, vortex_th)) + [(el[0], 0.) for el in exp_confs_from_outside_cha]
        vortex_th = sorted(vortex_th, key = lambda x: x[0])
        common_horizontal_axis_v_th = [el[0] for el in vortex_th]
        vortex_th = [el[1] for el in vortex_th]
        assert common_horizontal_axis_v == common_horizontal_axis_v_th
    #finally, we update common_horizontal_axis
    common_horizontal_axis = common_horizontal_axis_v

    assert len(common_horizontal_axis) == len(vortex)
    if MTD_th is not None:
        assert len(common_horizontal_axis) == len(vortex_th)


    if not np.isclose(sum(proportions)+sum(vortex), 1., atol=len(vortex)*1e-03):
        warn("""In estimate_proportions:
Proportions of signal and noise sum to %f instead of 1.
This may indicate improper results.
Please check the deconvolution results and consider reporting this warning to the authors.
                        """ % (sum(proportions)+sum(vortex)))
        
    compare_area = ((not nmr) or (nmr and what_to_compare=='area'))

    if compare_area:
        if MTD_th is not None:
            result_dict = {'proportions': proportions, 'noise': vortex, 'noise_in_components': vortex_th, 
                'proportion_of_noise_in_components': p0_prime, 'common_horizontal_axis': common_horizontal_axis, 
                   'Wasserstein distance': objective_function, 'output_warm_start_values': output_warm_start_values}
        else:
            result_dict = {'proportions': proportions, 'noise': vortex, 'common_horizontal_axis': common_horizontal_axis,
                    'Wasserstein distance': objective_function, 'output_warm_start_values': output_warm_start_values}
    else:
        queries_protons = [query_spec.protons for query_spec in preprocessed_query]
        rescaled_proportions = [prop/prot for prop, prot in zip(proportions, queries_protons)]
        rescaled_proportions = [prop/sum(rescaled_proportions) for prop in rescaled_proportions]
        result_dict = {'proportions': rescaled_proportions, 'Wasserstein distance': objective_function,
                        'output_warm_start_values': output_warm_start_values}

    return result_dict



def estimate_proportions_in_time(mixture_in_time, reagents_spectra, MTD=0.5, MDC=1e-8,
                                MMD=-1, max_reruns=3, verbose=False,
                                MTD_th=0.5, solver=lp.GUROBI(msg=False, warmStart=True),
                                what_to_compare='area'):

    """
    Returns estimated proportions of reagents and noise in mixture changing over time.
    Uses estimation from previous time point to speed up the computations.
    _____
    Parameters:
    mixture_in_time: list of Spectrum objects or np.ndarray
        The mixture's spectrum in consecutive time points. If list of Spectrum objects, then the earliest spectrum should be 
        an element 0 of the list. If np.ndarray, then column 0 of the array should contain chemical shift values, and the 
        other columns should contain intensities (those corresponding to the earliest spectrum should be in column 1).
    reagents_spectra: list of Spectrum objects
        A list of reagents' spectra (both substrates and products) present in the mixture.
    MTD: Maximum Transport Distance, float
        Signal from mixture's spectrum will be transported up to this distance when estimating
        reagents' proportions. This parameter is interpreted as denoising penalty for mixture.
        To disable denoising, set this parameter to large value (for example 1000). Default is 0.1. 
    MDC: Minimum Detectable Current, float
        If the spectrum of a reagent encompasses less than
        this amount of the total signal, it is assumed that this reagent
        is absent in the spectrum. Default is 1e-8.
    MMD: Maximum Mode Distance, float
        If there is no mixture's peak within this distance from the
        highest peak of spectrum of a reagent,
        it is assumed that this reagent is absent in the spectrum.
        Setting this value to -1 disables filtering. Default is -1.
    max_reruns: int
        Due to numerical errors, some partial results may be inaccurate.
        If this is detected, then those results are recomputed for a maximal number of times
        given by this parameter. Default is 3.
    verbose: bool
        Print diagnostic messages? Default is False.
    MTD_th: Maximum Transport Distance for reagents' spectra, float
        If presence of noise in reagents' spectra is not expected, 
        then this parameter should be set to None. Otherwise, set its value to some positive real number.
        Signal from reagents' spectra will be transported up to this distance 
        when estimating reagents' proportions. This parameter is interpreted as denoising penalty for reagents. Default is 1.0.
    solver: 
        Which solver should be used. We recommend using lp.GUROBI() (note that it requires obtaining a licence).
        To see all solvers available at your machine execute: pulp.listSolvers(onlyAvailable=True) or lp.listSolvers(onlyAvailable=True). 
        If you prefer to use Magnetstein without Gurobi set this argument to solver=LpSolverDefault.
        Note that using Magnetstein without Gurobi can result in long computation time and, in some cases, incorrect results.
    what_to_compare:
        Should the resulting proportions correspond to concentrations or area under the curve? Default is
        'concentration'. Alternatively can be set to 'area'. This argument is used only for NMR spectra.
    _____
    Returns: dict
        A dictionary with the following entries:
        - proportions_in_time: List of proportions of reagents' spectra changing in time.
        If what_to_compare='area' and MTD_th parameter is not equal to None, then the dictionary contains also 
        the following entries:
    """
        
        
    # Checking type of spectra

    are_reagents_NMR_spectra = [isinstance(sp, NMRSpectrum) for sp in reagents_spectra]
    assert all(are_reagents_NMR_spectra) or not any(are_reagents_NMR_spectra), \
            'Provided spectra of reagents are of mixed types. \
            Please assert that either all or none of the spectra are NMR spectra.'

    nmr = all(are_reagents_NMR_spectra)

    
    # Loading spectra of mixtures into a single list
    
    if isinstance(mixture_in_time, list):

        mixture_in_time_list = mixture_in_time

        are_mixtures_NMR_spectra = [isinstance(sp, NMRSpectrum) for sp in mixture_in_time_list]

        assert all(are_mixtures_NMR_spectra) or not any(are_mixtures_NMR_spectra), \
                'Provided spectra of mixtures are of mixed types. \
                Please assert that either all or none of the spectra are NMR spectra.'


    elif isinstance(mixture_in_time, np.ndarray):

        horizontal_axis = mixture_in_time[:,0]
        intensities = [mixture_in_time[:,i] for i in range(1, mixture_in_time.shape[1])]
        mixtures_confs_list = [list(zip(horizontal_axis, intensity)) for intensity in intensities]

        if nmr:
            mixture_in_time_list = [NMRSpectrum(confs=conf) for conf in mixtures_confs_list]
        else:
            mixture_in_time_list = [Spectrum(confs=conf) for conf in mixtures_confs_list]


    else:
        print('Cannot retrieve spectra of mixtures from mixture_in_time.\
                \n Make sure that provided object is either a list of Spectum objects or numpy.ndarray.')
        return


    
    # Preparing lists for storing the results

    proportions_in_time = []
    noise_proportions_in_time = []
    noise = []
    noise_in_reagents = []
    common_horizontal_axis_list = []

    
    # Estimation

    for i, mix in enumerate(mixture_in_time_list):
        if verbose:
            print('Analyzing timepoint '+str(i)+'.\n')

        current_mix = mix
        current_mix.trim_negative_intensities()
        current_mix.normalize()
        current_horizontal_axis = [conf[0] for conf in current_mix.confs]

        init_solver = deepcopy(solver)

        if i==0:

            estimation = estimate_proportions(current_mix, reagents_spectra, MTD=MTD, MDC=MDC,
                                                MMD=MMD, max_reruns=max_reruns, verbose=verbose,
                                                progress=False, MTD_th=MTD_th, solver=init_solver,
                                                what_to_compare=what_to_compare,
                                                warm_start_values=None
                                                )

        else:

            if current_horizontal_axis == previous_horizontal_axis:

                estimation = estimate_proportions(current_mix, reagents_spectra, MTD=MTD, MDC=MDC,
                                                MMD=MMD, max_reruns=max_reruns, verbose=verbose,
                                                progress=False, MTD_th=MTD_th, solver=init_solver,
                                                what_to_compare=what_to_compare,
                                                warm_start_values=current_warm_start_values
                                                )

            else:

                if verbose:
                    print('Current mixture spectrum (' + '+str(i)' + \
                            ') has different chemical shift axis than the previous one.\
                            Therefore, estimation for this spectrum will be performed \
                            without using information from the previous time point.')
                estimation = estimate_proportions(current_mix, reagents_spectra, MTD=MTD, MDC=MDC,
                                                MMD=MMD, max_reruns=max_reruns, verbose=verbose,
                                                progress=False, MTD_th=MTD_th, solver=init_solver,
                                                what_to_compare=what_to_compare,
                                                warm_start_values=None
                                                )


        previous_horizontal_axis = current_horizontal_axis

        proportions_in_time.append(estimation['proportions'])
        noise_proportions_in_time.append(estimation['proportion_of_noise_in_components'])
        noise.append(estimation['noise'])
        noise_in_reagents.append(estimation['noise_in_components'])
        common_horizontal_axis_list.append(estimation['common_horizontal_axis'])

        current_warm_start_values = deepcopy(estimation['output_warm_start_values'])

        if verbose:

            print('Proportions:\n')
            print(estimation['proportions'])
            print('\n')

    
    return {'proportions_in_time' : proportions_in_time,
            'noise_in_mixture_in_time' : noise, 
           'noise_in_reagents_in_time' : noise_in_reagents, 
            'proportion_of_noise_in_reagents_in_time': noise_proportions_in_time,
           'common_horizontal_axis_in_time' : common_horizontal_axis_list}