import patsy
import mpmath
from tqdm import tqdm
from functools import partial

import statsmodels.api as sm


class TweedieDifferentialExpression:

    def __init__(self, formula: str):
        assert(formula.split("~")[0].rstrip() == 'exog', "Must call the target vector 'exog'.")
        self.results = None
        self.formula = formula
        self.fitted = False

    @staticmethod
    def tweedie_log_likelihood_probability(glm_model, p):
        return -tweedie(mu=glm_model.mu, p=p, phi=glm_model.scale).logpdf(glm_model._endog).sum()

    @staticmethod
    def iterative_tweedie_estimate(endog, exog, n_iterations: int = 1):
        res = sm.GLM(endog, exog, family=sm.families.Tweedie(var_power=1.1)).fit()

        for _ in range(n_iterations):
            try:
                llf = partial(tweedie_log_likelihood_probability, res)
                opt = sp.optimize.minimize_scalar(llf, bounds=(1+1e-3, 2-1e-3), method='bounded')
                res = sm.GLM(endog, exog, family=sm.families.Tweedie(var_power=opt.x)).fit()
            except:
                continue
        return res

    @staticmethod
    def t_to_p_value(t_statistic, degrees_of_freedom=1):
        """Convert t-statistic to two-sided p-value with arbitrary precision."""
        with mpmath.workdps(1000):  # Adjust the precision as needed
            p_value = 2 * (1 - mpmath.gammainc((degrees_of_freedom + 1) / 2, 0, abs(t_statistic) * mpmath.sqrt(degrees_of_freedom / 2))) / mpmath.gamma(degrees_of_freedom / 2)
        return p_value

    def fit_tweedie_models(self, adata, min_expr = 0.05):
        # assert(~adata.obs.index.eq("exog").any(), "Found 'exog' in the adata.obs dataframe, please rename the column.")
        self.results = {}

        for feature in tqdm(adata[:, (adata.X>0).mean(axis=0) > min_expr].var.index):
            endog, exog = self.get_design_matricies(adata, feature)
            self.results[feature] = self.iterative_tweedie_estimate(endog, exog, n_iterations=1)
        
        self.fitted=True

        return self.results

    def get_design_matricies(self, adata, feature):
        return patsy.dmatrices(
            self.formula, 
            adata.obs.assign(exog=adata[:, feature].X.todense())
        )

    def summary_table(self):
        assert(self.fitted, "Please fit the model before returning the results.")
        coef = pd.DataFrame({
            ix: self.results[ix].params for ix in self.results
        }, index=["coef " + x for x in self.results[list(self.results.keys())[0]].summary2().tables[1].index]).T

        ttest = pd.DataFrame({
            ix: self.results[ix].tvalues for ix in self.results
        }, index=["t-test " + x for x in self.results[list(self.results.keys())[0]].summary2().tables[1].index]).T

        chi2 = pd.DataFrame({
            ix: self.results[ix].pearson_chi2 for ix in self.results
        }, index=["X^2"]).T

        pval = ttest.applymap(t_to_p_value)
        pval.columns = [x.replace("t-test", "p-value") for x in ttest.columns]

        padj = pval.copy() * pval.shape[0] * (pval.shape[1] - 1)
        padj.columns = [x.replace("p-value", "padj")  for x in pval.columns]

        log10p = padj.map(lambda x: -mpmath.log(x))
        log10p.columns = [x.replace("p-value", "log10p")  for x in pval.columns]

        df = pd.concat([coef, ttest, pval, padj, log10p, chi2], axis=1)

        return df

# Example Usage
microglia_differential_expression = TweedieDifferentialExpression("exog ~ 1 + cov1 + cov2")
microglia_differential_expression.fit_tweedie_models(adata, min_expr = 0.05)
df = microglia_differential_expression.microglia_differential_expression()