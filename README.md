# feature-filter
Introduction

Provide methods to filter or select redundant features. 
Support simple method to filter constant features, duplicated features, highly correlated features. 
Also support scenarios depended filters. Filter low information ratio for logistic regression model; filt low weight or low important features from models as random forest, logistic regression and linear_regression.

Key Public APIS
filter_auto() provides integrated API to Filter or select features with multi methods as infinite confine, Filter constant features, Filter duplicated features, Filter highly correlated features. This API also Filter feature according to weight from from different models as linear models and random forest.
filter_constant() filter features which contain single value or all value is nan. 
filter_duplication() filter duplicated features, support numeric type feature only.
filter_hi_corr() filter highly correlated features, support numeric type feature only.
filter_info_ratio() filter features with low information ratio, support category type feature. In credit predict, numeric type features are often binned for stable prediction and explainable.
filter_forest_importance()/ filter_LBGM_importance() filter features with low weight when estimate with random forest/LGBM model. More features are often filtered with filter_LGBM_importance().
filter_logistic_regression() / filter_linear_regression() filter features with low weight when estimate as logistic regression / linear regression model with LASSO regularization.

Usage

setEnvInfo() is necessary to setup log info path, before call functions to impute missing values. Variable debug is the switcher of debug info, while trace info is always output. Some constant values are defined as default algorithm parameter. They can be tuned if necessary, with assistant of log info.
