
	
	global pathmodel  "C:\Users\Sergi\Dropbox\PhD\Projects\Papers\1_financial_constraints\Model"
	
	
	use "${pathmodel}/dataepsilons", clear
	
	
	*------------------------------------------------------------------------*
	
	// Adjust budget (since now includes epslons)
	
	replace budget1 = budget1 - epsilon1 - debt_choice
	replace budget2 = budget2 - epsilon2 - debt_choice
	
	drop budget2
	rename budget1 budget
	
	* Try maximum likelihood of a truncated normal distribution
	
	gen newdebt = debt_choice - debt
	
	sum newdebt, d
	
	sum newdebt if newdebt > 0 & educ == 2, d
	
	replace newdebt = 20400 if newdebt > 20400
	
	gen havedebt = debt > 0 
	
	
	
	truncreg  newdebt i.field budget i.work i.sex i.race havedebt if educ == 2 & parinc == 1 & ability == 1 & fouryear_exp == 0 ,  ll(0) ul(20400)
	
cd ${modeldata}

python 
from sfi import Matrix
from sfi import Scalar
import numpy as np
import os

# Get the working directory
path = Macro.getGlobal('c(pwd)')
print(path)
os.chdir(f"{path}")

# Where sd will be stored
sigmas = []
stata: qui truncreg  newdebt i.field budget i.work i.sex i.race havedebt if educ == 2 & parinc == 1 & ability == 1 & fouryear_exp == 0 ,  ll(0) ul(20400)
betas= np.array(Matrix.get('e(b)'))
# The problem is that base categories are stored as 0.0 coefficients
betas = betas[betas!=0]
print(betas)
#np.save("function_coefficients/param_coltype_educ2.npy",betas)
end
	


	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	/*
	gen logdebt = log(debt_choice+1)
	
	
	
	gen havedebt = debt> 0 
	
	
	reg logdebt i.field budget1 i.work i.period i.sex i.race havedebt i.twoyear_exp if educ == 2 & parinc == 1 & ability == 1 & debt_choice > 0 
	predict debts
	gen expdebt = exp(debts)
	sum expdebt if educ == 2 & parinc == 1 & ability == 1 & debt_choice > 0 , d
	sum debt_choice if  educ == 2 & parinc == 1 & ability == 1 & debt_choice > 0, d
	
	truncreg  debt_choice i.field budget1 i.work i.period i.sex i.race havedebt i.twoyear_exp if educ == 2 & parinc == 1 & ability == 1 ,  ll(0) 
	predict debthat
	
	replace debthat = 0 if debthat < 0 
	
	sum debthat if educ == 2 & parinc == 1 & ability == 1, d
	sum debt_choice if educ == 2 & parinc == 1 & ability == 1, d
	
	predict debthat
	sum debthat if educ == 2 & parinc == 1 & ability == 1
	sum debt_choice if educ == 2 & parinc == 1 & ability == 1
	
	sum debthat if educ == 2 & parinc == 1 & ability == 1 & fouryear_exp == 0
	sum debt_choice if educ == 2 & parinc == 1 & ability == 1 & fouryear_exp == 0
	
	tobit debt_choice i.field budget1 i.work i.period i.sex i.race if educ == 2 & parinc == 1 & ability == 1, ll(0)
	predict double xb, ystar(0,1000000000000)
	scatter xb debt_choice if educ == 2 & parinc == 1 & ability == 1
	
	sum xb if educ == 2 & parinc == 1 & ability == 1
	sum debt_choice if educ == 2 & parinc == 1 & ability == 1
	
	
	gen newdebt = debt_choice - debt
	
	tobit newdebt i.field budget1 i.work i.period i.sex i.race if educ == 2 & parinc == 1 & ability == 1, ll(0)
	predict yhat
	scatter newdebt yhat
	
	gen anydebt = newdebt > 0 
	
	logit anydebt i.field debt i.work i.period i.sex i.race if educ == 2 & parinc == 1 & ability == 1
	
		
	
	predict debt_hat
	
	scatter debt_choice debt_hat if educ == 2 & parinc == 1 & ability == 1
	
	kdensity debt_choice if educ == 2 & parinc == 1 & ability == 1
	
	kdensity debt_hat if educ == 2 & parinc == 1 & ability == 1
	
	
	
	
	
	
	
	
	
	
	/*
	

	// Identify the minimum epsilon that does not generate debt at the x1 level
	
	gen nodebtchoice = debt == debt_choice
	bys parinc ability sex race educ nodebtchoice period: egen minepsi = min(epsilon1)
	gen temp = minepsi if nodebtchoice == 1
	bys  parinc ability sex race educ period: egen miniepsireal = max(temp)
	
	
	
	// Play Around
	
	replace epsilon1 = epsilon1 / 10000
		
	reg epsilon1 i.parinc i.ability i.sex i.race i.period i.educ i.work budget1 debt if epsilon1 > -20000 & epsilon1 < 20000 
	
	
	predict e1hat
	scatter e1hat epsilon1  if epsilon1 > -20000 & epsilon1 < 20000
	
	
	gen anydebt = debt_choice != debt
	
	logit anydebt i.parinc i.ability i.sex i.race i.period i.educ i.work  debt budget1
	
	margins, at(parinc == 1 ability == 1 period == 1 educ == 2)
	tab anydebt if ability == 1 & parinc == 1 & period == 1 & educ == 2
	
	reg epsilon1 budget1 debt i.period if parinc== 1 & ability == 1 & sex == 1 & race == 1
	
	
	gen debtnew = debt_choice - debt
	
	reg debtnew i.parinc i.ability i.sex i.race i.period i.educ i.work  debt budget1 if anydebt == 1
	
	reg anydebt i.parinc i.ability i.sex i.race i.period i.educ i.work  budget1 miniepsireal
	
	reg anydebt i.parinc i.ability i.sex i.race i.period i.educ i.work
	
	
	
	sum epsilon1 if parinc == 1 & ability == 1 & sex == 1 & race == 1 & period == 1 & educ == 1 & work == 1
	
	replace debt_choice = debt_choice/1000
	sum debt_choice if parinc == 1 & ability == 1 & sex == 1 & race == 1 & period == 1 & educ == 1 & work == 1
	
	reg epsilon1 c.budget1##c.budget1##c.budget1##c.budget1 if parinc == 1 & ability == 1 & race == 1 & sex == 1 & period == 1
	
	reg epsilon1 c.budget1##c.budget1##c.budget1 if parinc == 4 & ability == 2 & sex == 1 & race == 1 & period == 2
	
	reg debtnew c.budget1##c.budget1##c.budget1 debt if parinc == 4 & ability == 2 & sex == 1 & race == 1 & period == 2
	
	logit anydebt c.budget1##c.budget1##c.budget1 debt if parinc == 4 & ability == 2 & sex == 1 & race == 1 & period == 2
	
	logit anydebt c.budget1##c.budget1##c.budget1 i.parinc##i.ability##i.sex##i.race##i.period debt 
	margins, at(parinc == 1 ability==1  period == 2)
	tab anydebt if parinc == 1 & ability == 1 & period == 2
	
	
	preserve
	keep if epsilon1 < 50000
	graph twoway scatter e1hat epsilon1 || lfit e1hat epsilon1 
	restore
	
	reg epsilon2 i.parinc i.ability i.sex i.race i.period i.educ
	
	
	reg epsilon1 budget1 debt i.period i.educ i.work
	
	
	tab nodebtchoice
	
	table nodebtchoice, stat(mean epsilon1)  stat(mean epsilon2) nototals
	
	table nodebtchoice, stat(mean epsilon1)  stat(mean epsilon2) nototals
	
	table parinc nodebtchoice if educ == 1, stat(mean epsilon1)   nototals
	table parinc nodebtchoice if educ == 2, stat(mean epsilon1)   nototals
	
	
	
	*----------------------------------------------------------------------*
	
	// Identify the minimum epsilon that does not generate debt at the x1 level
	
	
	bys parinc ability sex race educ nodebtchoice period: egen minepsi = min(epsilon1)
	gen temp = minepsi if nodebtchoice == 1
	bys  parinc ability sex race educ period: egen miniepsireal = max(temp)
	
	tab minepsi if nodebtchoice == 1 & parinc == 4 & ability == 4 & sex == 0 & race == 0 
	tab miniepsireal if parinc == 4 & ability == 4 & sex == 0 & race == 0 
	

	
	
	gen bigger = epsilon1 > miniepsireal
	
	tab nodebtchoice if parinc == 1 & ability == 4 & sex == 0 & race == 0 & educ == 2 & period == 2
	tab bigger if parinc == 1 & ability ==4  & sex == 0 & race == 0 & educ == 2 & period == 2
	
	
	*-------------------------------------------------------------------------*
	
	// Fit a Tobit on debt
	
	
	replace debtnew = debtnew /10000
	replace budget1 = budget1/10000
	
	tobit debtnew budget1  i.race i.sex i.parinc i.ability i.fouryear_exp if educ == 2, ll(0)
	margins, predict (ystar(0,.))

	
	sum debtnew if parinc == 1 & ability == 1 & educ == 2 &   race == 1 & sex == 1  & debtnew > 0 
	
	help tobit
	
	
	sum epsilon1 if parinc == 4 & ability == 1 & educ == 2 & period == 6