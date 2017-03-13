# hubblefit
Tool to get the standardization parameters for Type Ia Supernova Hubble Diagram fit. (no cosmo fit yet)

# Usage

### The Data format

You need a dictionary (here `data`) with the following structure:
```
data = {name: SN_DATA_INFO}
```

SN_DATA_INFO is a dictionary containing: 
- `mag`: The observed SN magnitude
- `zcmb`: The redshift of the SN in the cmb frame

In addition it can have any parameter that you can later whant to use as standardization coefficient. 
For instance, if you want to standardize using x1 and c you need to add them to SN_DATA_INFO.

#### Errors
The parameter error must be register as parameter_name.err (e.g. for `x1`, `x1.err`)

If the parameter_name.err is not found, the error is assumed to be 0.
#### Covariance
The covariance between two coefficients p1, p2 could be given as `cov_p1p2` (e.g., between `mag` and `c`, `cov_magc` or `cov_cmag`)

If the covariance between two parameters is not found, it is assumed to be 0.

### Fitting the data
You standardize using any parameter given in the `data`. Set that in `corr`
```
import hubblefit
hfit = hubblefit.get_hubblefit(data, corr=[p1,p2])
hfit.fit()
```

The best fitted values are accessible as `hfit.fitvalue`

# Dependencies
- modefit (`pip install modefit`)
