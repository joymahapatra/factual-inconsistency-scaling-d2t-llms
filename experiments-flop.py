import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter, MultipleLocator, AutoMinorLocator
from scipy.optimize import curve_fit
import pandas as pd
from scipy.stats import norm, f
from scipy.stats import shapiro
from sklearn import model_selection
import utils


# plt.rcParams.update({'font.size': 8})
TINY_SIZE = 6
SMALL_SIZE = 6
MEDIUM_SIZE = 8
BIGGER_SIZE = 10
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=TINY_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TINY_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=TINY_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
mpl.rcParams['axes.linewidth'] = 0.01
plt.locator_params(axis='y', nbins=10)
mpl.rcParams['text.usetex'] = True


def plot_regression_with_error(x, y, residuals, axs, color='red', name_scaling_law:str=None, scaling_param:float=None):
    """
    Plot the fitted regression line with error margins.
    """
    scaling_law_linestyles = {'exp_scaling_law': 'solid', 'pow_scaling_law': 'dashed'} # "solid", "dotted", "dashed"
    linestyle = scaling_law_linestyles[name_scaling_law] 
    actual_names = {'exp_scaling_law': 'Exponential', 'pow_scaling_law': 'Power law'}
    # Compute standard error of predictions
    se = np.std(residuals) / np.sqrt(len(residuals))
    margin = 1.96 * se  # 95% confidence interval
    if name_scaling_law == "exp_scaling_law":
        # plot_label = fr"{actual_names[name_scaling_law]} scaling ($\beta$={scaling_param:1.2f})"
        plot_label = fr"{actual_names[name_scaling_law]} scaling"
        axs.plot(x, y, color='b', label=plot_label, linestyle=linestyle, linewidth=0.75)
        fill_label = f"Margin of error (exponential scaling)"
        axs.fill_between(x, y - margin, y + margin, color='b', alpha=0.25, linewidth=0.2, label=fill_label) # 'darkturquoise'
    else:
        # plot_label = fr"{actual_names[name_scaling_law]} scaling ($\alpha$={scaling_param:1.2f})"
        plot_label = fr"{actual_names[name_scaling_law]} scaling"
        axs.plot(x, y, color='r', label=plot_label, linestyle=linestyle, linewidth=0.75)
        fill_label = f"Margin of error (power law scaling)"
        axs.fill_between(x, y - margin, y + margin, color='r', alpha=0.25, linewidth=0.2, label=fill_label) # 'lightgrey'

def vuong_test(log_likelihoods_model1, log_likelihoods_model2, alpha=0.05):
    """
    log_likelihoods_model1: list of log-likelihoods for model 1.
    log_likelihoods_model2: list of log-likelihoods for model 2.
    alpha: Significance level for the test.
    """
    n = len(log_likelihoods_model1)
    lr = log_likelihoods_model1 - log_likelihoods_model2
    lr_mean = np.mean(lr)
    lr_var = np.var(lr, ddof=1)  # Sample variance
    test_statistic = lr_mean * np.sqrt(n / lr_var)
    p_value = 2 * (1 - norm.cdf(abs(test_statistic)))
    if test_statistic > norm.ppf(1 - alpha / 2):
        decision = "Model 1 preferred"
    elif test_statistic < -norm.ppf(1 - alpha / 2):
        decision = "Model 2 preferred"
    else:
        decision = "No significant difference"
    return test_statistic, p_value, decision

def shapiro_wilk_test(data):
    """
    data: A 1D numpy array or list of data.
    """
    statistic, p_value = shapiro(data)
    alpha = 0.05  # Significance level (you can adjust this)
    if p_value > alpha:
        interpretation = "Data looks normally distributed (fail to reject H0)"
    else:
        interpretation = "Data does not look normally distributed (reject H0)"
    return statistic, p_value, interpretation

def f_test_linear_regression(x, y, y_pred):
    """
    Performs an F-test for the goodness of fit of a linear regression model.
    """
    n = len(x)  # Number of data points
    p = 2  # (rate/exponent and intercept)
    residuals = y - y_pred
    ssr = np.sum(residuals**2)  # Residual sum of squares (RSS)
    sst = np.sum((y - np.mean(y))**2)  # Total sum of squares (SST)
    ssr_reg = sst - ssr  # Regression sum of squares (SSR_reg)
    f_statistic = (ssr_reg / p) / (ssr / (n - p -1))
    p_value = 1 - f.cdf(f_statistic, p , n - p -1)
    alpha = 0.05
    if p_value < alpha:
        decision = f"Reject the null hypothesis. The linear regression model is statistically significant."
    else:
        decision = f"Fail to reject the null hypothesis. The linear regression model may not be significant."
    return f_statistic, p_value, decision

def huber_loss(y_true, y_pred, delta=1.345):
  """
  the Huber loss function.
  """
  error = y_true - y_pred
  loss = np.where(np.abs(error) <= delta, 
                  0.5 * error**2, 
                  delta * (np.abs(error) - 0.5 * delta))
  loss = np.mean(loss).item()
  return loss

def perdictive_performance(scaling_type, xdata, ydata):
    """5-fold cross validation"""
    # print(*popt, pconv)
    all_loss = []
    for i_fold in range(5):
        xtrain, xtest, ytrain, ytest = model_selection.train_test_split(xdata, ydata, test_size=0.20)
        popt, pconv = curve_fit(scaling_type, xtrain, ytrain, p0=[1.0, 1.0, 0.1], maxfev=5000, method='trf', loss='huber')
        y_pred = scaling_type(xtest, *popt)
        all_loss.append(huber_loss(ytest, y_pred))
    all_loss = np.array(all_loss)
    mean_loss = np.mean(all_loss).item()
    std_loss = np.std(all_loss).item()
    return mean_loss, std_loss

# Exponential scaling
def exp_scaling_law(x, rate, constant, intercept):
    return (np.exp(rate * x) * constant) + intercept

# Power law scaling
def pow_scaling_law(x, exponent, constant, intercept):
    return (constant * (x ** exponent)) + intercept


def find_log_likelihood(y_true, y_pred):
    """likelihood"""
    residuals = y_true - y_pred
    # Calculate log-likelihood
    n = len(y_true)
    sigma_squared = np.var(residuals)
    log_likelihood = -0.5 * n * np.log(2 * np.pi * sigma_squared) - 0.5 * (residuals**2 / sigma_squared)
    return log_likelihood, residuals



if __name__ == "__main__":
    """main"""
    """data collecting"""    
    entire_data = pd.read_csv("./input_size_scaling.tsv", sep='\t', index_col=None, header='infer', encoding="utf-8")
    
    dataset_name_list = ['DART', 'E2E', 'ViGGO', 'WebNLG', 'WikiTableText']
    # dataset_color_list = ['blue', 'red', 'indigo', 'green', 'brown']
    dataset_color_list = ['black'] * len(dataset_name_list)
    all_metrics = ["AlignScore", "QAFactEval", "SummaC-conv", "UniEval-fact", "Human"]
    all_models = ["BLOOM", "FLAN-T5", "Mamba", "OPT", "Pythia"]
    metric_id = int(input("Enter the metric ID. > "))
    metric_name = all_metrics[metric_id]

    """storage allocation"""
    os.makedirs("./flop_scaling_results", exist_ok=True)
    
    for i_model in all_models:
        t_model_name = i_model.lower().replace("-", "")
        global_tag = (i_model + "_" + metric_name).lower()
        all_store = utils.Store2DResult(name=f"{global_tag}_all_store", store_dir_path="flop_scaling_results")
        data = entire_data[entire_data['Model'].str.startswith(t_model_name)]
        if len(data) == 0:
            print(f"No entry for {i_model} family.")
            continue
        given_datasets = list(set(data.columns).intersection(set(dataset_name_list)))
        given_datasets.sort()
        num_datasets = len(given_datasets)
        fig, axs = plt.subplots(nrows=1, ncols=len(given_datasets), figsize=(8, 1.75))
        for idx, dataset_name_t in enumerate(given_datasets):
            xdata = data['Size'+ "-" + dataset_name_t].to_numpy()
            xdata = xdata + 1e-10 # prevent underflow
            ydata = data[dataset_name_t].to_numpy()
            axs[idx].scatter(x=xdata, y=ydata, marker=r'o', c='black', s=5) # dataset_color_list[idx]
            all_predictions = {}
            all_likelihoods = {}
            for scaling_type in [exp_scaling_law, pow_scaling_law]:
                print(f"We are with {dataset_name_t}, {t_model_name} and {scaling_type.__name__}.")
                # popt, pconv = curve_fit(scaling_type, xdata, ydata, p0=[1.0, 1.0, 0.10], maxfev=5000, method='trf', loss='huber')
                popt, pconv = curve_fit(scaling_type, xdata, ydata, p0=[1.0, 1.0, 0.1], maxfev=5000, method='trf', loss='huber')
                all_store.data[dataset_name_t][f"rate/expn_{scaling_type.__name__}"] = f"{popt[0]:1.2e}"
                all_store.data[dataset_name_t][f"constant_{scaling_type.__name__}"] = f"{popt[1]:1.2e}"
                all_store.data[dataset_name_t][f"intercept_{scaling_type.__name__}"] = f"{popt[2]}"
                # print(*popt, pconv)
                all_predictions[scaling_type.__name__] = scaling_type(xdata, *popt)
                # plt.plot(xdata.tolist(), all_predictions[scaling_type.__name__].tolist(), label=scaling_type.__name__)
                """stage I: predictive performance estimation"""
                pp_mean, pp_std = perdictive_performance(scaling_type=scaling_type, xdata=xdata, ydata=ydata)
                all_store.data[dataset_name_t][f"mean_error_{scaling_type.__name__}"] = f"{pp_mean:1.2e}"
                all_store.data[dataset_name_t][f"std_error_{scaling_type.__name__}"] = f"{pp_std:1.2e}"
                """Stage II: Goodness-of-fit"""
                """<start f-test>"""
                if scaling_type.__name__ == "exp_scaling_law":
                    x_ftest = xdata
                    y_ftest = ydata -  popt[-1]
                    y_ftest = np.log(y_ftest)
                    y_pred_ftest = all_predictions[scaling_type.__name__] -  popt[-1]
                    y_pred_ftest = np.log(y_pred_ftest)
                    _, f_test_p, _ = f_test_linear_regression(x=x_ftest, y=y_ftest, y_pred=y_pred_ftest)
                elif scaling_type.__name__ == "pow_scaling_law":
                    x_ftest = np.log(xdata)
                    y_ftest = ydata -  popt[-1]
                    y_ftest = np.log(y_ftest)
                    y_pred_ftest = all_predictions[scaling_type.__name__] -  popt[-1]
                    y_pred_ftest = np.log(y_pred_ftest)
                    _, f_test_p, _ = f_test_linear_regression(x=x_ftest, y=y_ftest, y_pred=y_pred_ftest)
                else:
                    raise NotImplementedError
                all_store.data[dataset_name_t][f"ftest_pval_{scaling_type.__name__}"] = f_test_p
                """<end f-test>"""
                xdata_p = np.linspace(np.min(xdata), np.max(xdata), num=100)
                ydata_p = scaling_type(xdata_p, *popt)
                # plt.plot(xdata_p.tolist(), ydata_p.tolist(), label=scaling_type.__name__)
                all_likelihoods[scaling_type.__name__], residuals = find_log_likelihood(y_true=ydata, y_pred=all_predictions[scaling_type.__name__])
                _, shapiro_p, interpretation = shapiro_wilk_test(residuals)
                """assumption verificcation"""
                all_store.data[dataset_name_t][f"shwilk_pval_{scaling_type.__name__}"] = shapiro_p
                plot_regression_with_error(x=xdata_p, y=ydata_p, residuals=residuals,
                                        axs=axs[idx],
                                        color=dataset_color_list[idx],
                                        name_scaling_law=scaling_type.__name__,
                                        scaling_param=popt[0])
                # axs[idx].legend(["data", "exponential", "power law"], loc="upper right")
                # axs[idx].legend(loc="upper right")
                axs[idx].set_title(fr"\textbf{{{dataset_name_t} (w/ {i_model} Family)}}", loc='center')
                # axs[idx].axis("off")
                # axs[idx].set_yticks([])
                # axs[idx].set_yticklabels([])
                axs[idx].set_xlabel(r'FLOPs ($\times 10^{20}$) $\rightarrow$')
                axs[idx].grid(True, linestyle='dotted', color='black', alpha=0.1, which='both')
                axs[idx].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                axs[idx].locator_params(axis='y', nbins=11)
                # axs[idx].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                # fig.text(0.5, 0.04, 'common X', va='center', ha='center')
                # fig.text(0.04, 0.5, 'common Y', va='center', ha='center', rotation='vertical')
            """Stage III: comparative analysis"""
            _, vuong_p, _ = vuong_test(log_likelihoods_model1=all_likelihoods['exp_scaling_law'], log_likelihoods_model2=all_likelihoods['pow_scaling_law'])
            all_store.data[dataset_name_t][f"vuong_pval_{scaling_type.__name__}"] = vuong_p
        
        # <start>: for getting legened figure
        # lines = []
        # labels = []
        # for idx, ax in enumerate(fig.axes):
        #     if idx == 0:
        #         Line, Label = ax.get_legend_handles_labels()
        #         lines.extend(Line)
        #         labels.extend(Label)
        #     fig.delaxes(ax=ax)
        #     # break
        # fig.legend(lines, labels, loc='upper center', 
        #            bbox_to_anchor=(0.5, 1.1), ncol=4,
        #            #    fancybox=True, shadow=True,
        #            fontsize=MEDIUM_SIZE)
        # <end>
        
        # fig.supxlabel('')
        fig.supylabel(fr'Factual Inconsistency w/ \textsc{{{metric_name}}} $\rightarrow$', fontsize=SMALL_SIZE, x=0.01)
        fig.tight_layout()
        fig.savefig(f"flop_scaling_results/{global_tag}_plot.pdf", format="pdf", dpi=200, bbox_inches='tight') 
        """dumping all"""
        all_store.dump()