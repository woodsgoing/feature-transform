# feature-transform
Introduce
Provide methods to transform data like binning feature, polynomial feature, bias feature, feature aggregation along baseline.

Key Public APIS

transform_auto() provides integrated API, automatically handle infinite feature values, generate deviate variables from solo feature and pair features, binning both nominal and numeric features, and aggregate data with two dimensions at most.
confine_infinite() reassign feature value from infinite to it's boundary values.
binning() binning values for smooth, stable and descriptable prediction, especially for regression methods.
generate_on_solo_feature() generate more features with arithmatic rule from one feature.
generate_bias_features() calculate different values between similar features.
generate_polynomial_feature() generate derivations with pair numeric feautes or pair nominal features.
aggregrate_on_key() aggregate as various arithmatic methods(mean/ size/ min/ max/std) according to key feature which repeats within multi rows.
aggregrate_along_baseline() aggregate features along some continuous baseline which is segmented as aggregate key feature
aggregrate_on_key_along_baseline() aggregate according to key feature and seqmented baseline feature. It's always useful to analyze time series related features

Usage

setEnvInfo() is necessary to setup log info path, before call functions. Variable debug is the switcher of debug info, while trace info is always output. Some constant values are defined as default algorithm parameter. They can be tuned if necessary, with assistant of log info.
