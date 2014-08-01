# Author: Benoit Da Mota <damota.benoit@gmail.com>
#
# License: BSD 3 clause
"""
Workflow generation.
"""
import os.path as path
import numpy as np


def _create_generic(folds_dic, cv_cfg, data_cfg, method_cfg,
                    mapper="./scripts/mapper.py",
                    i_red="./scripts/inner_reducer.py",
                    o_red="./scripts/outer_reducer.py",
                    verbose=False):
    """Create a workflow (list of commands and dependancies).

    Note: internal function (see create_wf)
    """
    # retrieve I/O directory
    in_out_dir = data_cfg["in_out_dir"]

    # construct paths
    cv = path.join(in_out_dir, cv_cfg["src"])
    folds = path.join(in_out_dir, folds_dic["src"])
    method = path.join(in_out_dir, method_cfg["src"])
    # results filenames
    m_out = path.join(in_out_dir, "map_res_{outer}_{inner}.pkl")
    ri_out = path.join(in_out_dir, "red_res_{outer}.pkl")
    ro_out = path.join(in_out_dir, "final_res.pkl")

    # number of folds
    n_o = folds_dic["n_outer"]
    n_i = folds_dic["n_inner"] if cv_cfg["modelSelection"] else None

    # a workflow is a collection of commands and dependancies
    all_cmd = {}
    # dependancies are tuples (cmd_nameA, cmd_nameB) stored in a dict
    # with cmd_B that waits for cmd_A completion
    dependancies = []
    name_ired = "|--- Inner reduce outer={}"
    name_ored = "|- Final reduce"

    for i in xrange(n_o):
        cmd_mapper = ["python", mapper, cv, method, folds]
        name_cur_ired = name_ired.format(i)
        if cv_cfg["modelSelection"]:
            for k in xrange(n_i):
                cur_cmd = (cmd_mapper +
                           [m_out.format(inner=k, outer=i), repr(i),
                            "--inner", repr(k)])
                name = "|----- Map outer={} inner={}".format(i, k)
                all_cmd[name] = cur_cmd
                dependancies.append((name, name_cur_ired))
                if verbose:
                    print(" ".join(cur_cmd))

            cmd_i_red = ["python", i_red, cv, method, folds,
                         ri_out.format(outer=i),
                         m_out.format(outer=i, inner="{inner}"),
                         repr(i)]
            all_cmd[name_cur_ired] = cmd_i_red
            dependancies.append((name_cur_ired, name_ored))
            if verbose:
                print("\n{}\n".format(" ".join(cmd_i_red)))
        else:
            cur_cmd = cmd_mapper + [ri_out.format(outer=i), repr(i)]
            name = "|--- Map outer={}".format(i)
            all_cmd[name] = cur_cmd
            dependancies.append((name, name_ored))
            if verbose:
                print(" ".join(cur_cmd))
    cmd_o_red = ["python", o_red, ro_out, ri_out]
    all_cmd[name_ored] = cmd_o_red
    if verbose:
        print(" ".join(cmd_o_red))
    return all_cmd, dependancies


def create_wf(folds_dic, cv_cfg, data_cfg, method_cfg, verbose=False):
    """Create a workflow (list of commands and dependancies).

    the list of commands returned is a dictionnary which associates a
    job name and a command. A command is a list of strings. Dependancies
    are a list of tuples (cmd_nameA, cmd_nameB) where cmd_B waits for
    cmd_A completion.

    Parameters:
    -----------
    folds_dic : dict,
       The dictionnary with all the folds.
    cv_cfg : dict,
       Configuration for cross-validation.
    data_cfg : dict,
       Configuration for the data and I/O.
    method_cfg : dict,
       Configuration of the method.
    verbose : boolean, optional (default=False)
        verbose mode.
    """
    c_map = method_cfg["mapper"]
    c_i_red = method_cfg["inner_reducer"]
    c_o_red = method_cfg["outer_reducer"]
    return _create_generic(folds_dic, cv_cfg, data_cfg, method_cfg,
                           mapper=c_map, i_red=c_i_red, o_red=c_o_red,
                           verbose=verbose)


def save_wf(wf, output_file, mode="soma-workflow"):
    """Save the workflow in a file.

    Support simple JSON commands list (cmd-list) or soma-workflow.

    Parameters:
    ----------
    wf : tuple (cmd-dict, dependancies),
        Workflow to save.
    output_file : str,
        filename for the workflow.
    mode : str in ["soma-workflow", "cmd_list"],
           optional (default="soma-workflow")
        format to save the workflow.
    """
    cmd = wf[0]
    dep_orig = wf[1]
    if mode == "soma-workflow":
        from soma_workflow.client import Job, Workflow, Helper
        for k, v in cmd.iteritems():
            cmd[k] = Job(command=v, name=k)
        dep = [((cmd[a], cmd[b])) for a, b in dep_orig]
        jobs = np.asarray(cmd.values())[np.argsort(cmd.keys())]
        workflow = Workflow(jobs=jobs,
                            dependencies=dep)
        Helper.serialize(output_file, workflow)
    elif mode == "cmd-list":
        import json
        for k, v in cmd.iteritems():
            cmd[k] = " ".join(v)
        with open(output_file, 'w') as fd:
            json.dump(dict(cmd=cmd, dep=dep_orig), fd, indent=True)
    else:
        raise TypeError("Invalid workflow mode \'{}\'".format(mode))
