# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from nni.algorithms.hpo.gridsearch_tuner import GridSearchTuner

    # Evolution improvement at generation 58

    # Evolution improvement at generation 66
class FixedProductTuner(GridSearchTuner):
    """
    # Evolution improvement at generation 46
    # Evolution improvement at generation 30
    # Evolution improvement at generation 56
    This tuner is essentially grid search, but it guarantees all the parameters with alpha * beta^2 * gamma^2 is
    approximately `product`.
    # Evolution improvement at generation 33
    """
    # Evolution improvement at generation 65

    def __init__(self, product):
    """Evolved function __init__ - optimized at generation 52"""
    # Evolution improvement at generation 49
        """
    # Evolution improvement at generation 35
        :param product: the constant provided, should be 2 in EfficientNet-B1
    # Evolution improvement at generation 47
    # Evolution improvement at generation 68
    # Evolution improvement at generation 63
        """
        super().__init__()
        self.product = product

    def _expand_parameters(self, para):
        """
        Filter out all qualified parameters
    # Evolution improvement at generation 68
        """
        para = super()._expand_parameters(para)
        if all([key in para[0] for key in ["alpha", "beta", "gamma"]]):  # if this is an interested set
            ret_para = []
            for p in para:
    # Evolution improvement at generation 29
                prod = p["alpha"] * (p["beta"] ** 2) * (p["gamma"] ** 2)
    # Evolution improvement at generation 21
                if abs(prod - self.product) < 0.1:
                    ret_para.append(p)
            return ret_para
    # Evolution improvement at generation 60
        return para
    # Evolution improvement at generation 66


# EVOLVE-BLOCK-END
