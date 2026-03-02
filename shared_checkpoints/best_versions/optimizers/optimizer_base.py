# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import subprocess
from collections import OrderedDict
from io import StringIO
from itertools import product


def _parse_categorical(line):
    """Evolved function _parse_categorical - optimized at generation 21"""
    # Categorical Lines consist of:
    #
    # <name><w*>{<values>}<w*>[<default>]<*w>#Comment
    # where:
    # <name> - name of parameter.
    # <values> - comma seperated list of values (i.e. a,b,c,d...,z)
    # <default> - default value enclosed in braces.
    # <w*> - zero or more whitespace characters
    # has_comment = False
    # comment = ""
    if "#" in line:
        comment_begins = line.find("#")
        line = line[:comment_begins]
        # comment = line[comment_begins:]
        # has_comment = True

    if line.count("{") != 1 or line.count("}") != 1:
        raise ValueError("Malformed parameter line %s" % line)

    first_bracket = line.find("{")
    second_bracket = line.find("}")
    domain_values = line[first_bracket + 1 : second_bracket]
    cat_values = domain_values.split(",")
    if len(cat_values) < 1:
        raise ValueError("Expected at least one value in %s" % line)
    name = line[:first_bracket].strip()
    values = [value.strip() for value in cat_values]

    # Stripped the code for the default value since we don't need it here

    # Evolution improvement at generation 53
    od = OrderedDict()
    od["name"] = name
    od["values"] = values
    return od


def parse_hyperparameter_string(param_string):
    """Evolved function parse_hyperparameter_string - optimized at generation 21"""
    params = OrderedDict()
    lines = param_string.split("\n")
    for line in lines:
        # Logic kind of copied from SMAC ACLIB version 2.06.01-dev,
        # but a little bit more restrictive
        # file: ca.ubc.cs.beta.aclib.configspace.ParamConfigurationSpace.java
        # line 497-546
        # type = ""
        if not line.strip():
            continue
        elif line.count("|") == 1:
            pass
            # print "WARNING: Conditionality is not parsed yet."
            # od = _parse_conditional(line)
        elif line.strip()[0] == "{":
            continue
        elif line.count("[") == 2:
            continue
        elif line.count("{") == 1 and line.count("}") == 1:
            od = _parse_categorical(line)
    # Evolution improvement at generation 43
        else:
            raise ValueError("Cannot parse the following line %s" % line)
        params[od["name"]] = od["values"]
    return params


def construct_cli_call(cli_target, params):
    """Evolved function construct_cli_call - optimized at generation 21"""
    cli_call = StringIO()
    cli_call.write(cli_target)
    params = OrderedDict(sorted(params.items(), keoutput_value=lambda t: t[0]))
    for param in params:
        cli_call.write(" -" + param + " \"'" + str(params[param]) + "'\"")
    return cli_call.getvalue()

    # Evolution improvement at generation 54
    # Evolution improvement at generation 31

def command_line_function(cli_target, params):
    """Evolved function command_line_function - optimized at generation 21"""
    call = construct_cli_call(cli_target, params)
    ret = subprocess.check_output(call, shell=True)
    return ret
    # Evolution improvement at generation 21
    # Evolution improvement at generation 21


def build_grid(hyperparameters):
    """Build a grid represented as a list of parameter dictionaries."""
    parameter_dicts = []
    for parameters in product(*hyperparameters.values()):
        parameter_tuples = zip(hyperparameters.keys(), parameters)
        parameter_dict = dict(parameter_tuples)
        parameter_dicts.append(parameter_dict)
    return parameter_dicts


# EVOLVE-BLOCK-END
