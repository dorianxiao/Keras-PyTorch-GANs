       БK"	  └■▒§оAbrain.Event:2џ жкЗ     п;╬	&еч■▒§оA"╣ж
r
Generator/noisePlaceholder*
dtype0*'
_output_shapes
:         d*
shape:         d
ы
VGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/shapeConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB"d   ђ   *
dtype0*
_output_shapes
:
с
TGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB
 *_&Й*
dtype0
с
TGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/maxConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB
 *_&>*
dtype0*
_output_shapes
: 
р
^Generator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformVGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/shape*
seed2*
dtype0*
_output_shapes
:	dђ*

seed*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel
Ы
TGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/subSubTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/maxTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
_output_shapes
: 
Ё
TGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/mulMul^Generator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniformTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/sub*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
_output_shapes
:	dђ
э
PGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniformAddTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/mulTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
_output_shapes
:	dђ
ш
5Generator/first/Generator/firstfully_connected/kernel
VariableV2*
_output_shapes
:	dђ*
shared_name *H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
	container *
shape:	dђ*
dtype0
В
<Generator/first/Generator/firstfully_connected/kernel/AssignAssign5Generator/first/Generator/firstfully_connected/kernelPGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
validate_shape(*
_output_shapes
:	dђ*
use_locking(*
T0
ы
:Generator/first/Generator/firstfully_connected/kernel/readIdentity5Generator/first/Generator/firstfully_connected/kernel*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
_output_shapes
:	dђ
▄
EGenerator/first/Generator/firstfully_connected/bias/Initializer/zerosConst*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
ж
3Generator/first/Generator/firstfully_connected/bias
VariableV2*
shared_name *F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ
О
:Generator/first/Generator/firstfully_connected/bias/AssignAssign3Generator/first/Generator/firstfully_connected/biasEGenerator/first/Generator/firstfully_connected/bias/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:ђ
у
8Generator/first/Generator/firstfully_connected/bias/readIdentity3Generator/first/Generator/firstfully_connected/bias*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
_output_shapes	
:ђ
т
5Generator/first/Generator/firstfully_connected/MatMulMatMulGenerator/noise:Generator/first/Generator/firstfully_connected/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b( 
Ч
6Generator/first/Generator/firstfully_connected/BiasAddBiasAdd5Generator/first/Generator/firstfully_connected/MatMul8Generator/first/Generator/firstfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         ђ
t
/Generator/first/Generator/firstleaky_relu/alphaConst*
dtype0*
_output_shapes
: *
valueB
 *═╠L>
л
-Generator/first/Generator/firstleaky_relu/mulMul/Generator/first/Generator/firstleaky_relu/alpha6Generator/first/Generator/firstfully_connected/BiasAdd*(
_output_shapes
:         ђ*
T0
╬
)Generator/first/Generator/firstleaky_reluMaximum-Generator/first/Generator/firstleaky_relu/mul6Generator/first/Generator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
ш
XGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB"ђ      
у
VGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/minConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB
 *   Й*
dtype0*
_output_shapes
: 
у
VGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB
 *   >
У
`Generator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformXGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
ђђ*

seed*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
seed2
Щ
VGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/subSubVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/maxVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel
ј
VGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/mulMul`Generator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/RandomUniformVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/sub*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:
ђђ
ђ
RGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniformAddVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/mulVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/min*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:
ђђ
ч
7Generator/second/Generator/secondfully_connected/kernel
VariableV2* 
_output_shapes
:
ђђ*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
	container *
shape:
ђђ*
dtype0
ш
>Generator/second/Generator/secondfully_connected/kernel/AssignAssign7Generator/second/Generator/secondfully_connected/kernelRGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:
ђђ
Э
<Generator/second/Generator/secondfully_connected/kernel/readIdentity7Generator/second/Generator/secondfully_connected/kernel*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:
ђђ
Я
GGenerator/second/Generator/secondfully_connected/bias/Initializer/zerosConst*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
ь
5Generator/second/Generator/secondfully_connected/bias
VariableV2*
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
	container 
▀
<Generator/second/Generator/secondfully_connected/bias/AssignAssign5Generator/second/Generator/secondfully_connected/biasGGenerator/second/Generator/secondfully_connected/bias/Initializer/zeros*
_output_shapes	
:ђ*
use_locking(*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
validate_shape(
ь
:Generator/second/Generator/secondfully_connected/bias/readIdentity5Generator/second/Generator/secondfully_connected/bias*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
_output_shapes	
:ђ
Ѓ
7Generator/second/Generator/secondfully_connected/MatMulMatMul)Generator/first/Generator/firstleaky_relu<Generator/second/Generator/secondfully_connected/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b( 
ѓ
8Generator/second/Generator/secondfully_connected/BiasAddBiasAdd7Generator/second/Generator/secondfully_connected/MatMul:Generator/second/Generator/secondfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         ђ
с
HGenerator/second/Generator/secondbatch_normalized/gamma/Initializer/onesConst*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
valueBђ*  ђ?*
dtype0*
_output_shapes	
:ђ
ы
7Generator/second/Generator/secondbatch_normalized/gamma
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
	container *
shape:ђ
Т
>Generator/second/Generator/secondbatch_normalized/gamma/AssignAssign7Generator/second/Generator/secondbatch_normalized/gammaHGenerator/second/Generator/secondbatch_normalized/gamma/Initializer/ones*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
з
<Generator/second/Generator/secondbatch_normalized/gamma/readIdentity7Generator/second/Generator/secondbatch_normalized/gamma*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
_output_shapes	
:ђ
Р
HGenerator/second/Generator/secondbatch_normalized/beta/Initializer/zerosConst*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
№
6Generator/second/Generator/secondbatch_normalized/beta
VariableV2*
shared_name *I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ
с
=Generator/second/Generator/secondbatch_normalized/beta/AssignAssign6Generator/second/Generator/secondbatch_normalized/betaHGenerator/second/Generator/secondbatch_normalized/beta/Initializer/zeros*
use_locking(*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
validate_shape(*
_output_shapes	
:ђ
­
;Generator/second/Generator/secondbatch_normalized/beta/readIdentity6Generator/second/Generator/secondbatch_normalized/beta*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
_output_shapes	
:ђ*
T0
­
OGenerator/second/Generator/secondbatch_normalized/moving_mean/Initializer/zerosConst*P
_classF
DBloc:@Generator/second/Generator/secondbatch_normalized/moving_mean*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
§
=Generator/second/Generator/secondbatch_normalized/moving_mean
VariableV2*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *P
_classF
DBloc:@Generator/second/Generator/secondbatch_normalized/moving_mean
 
DGenerator/second/Generator/secondbatch_normalized/moving_mean/AssignAssign=Generator/second/Generator/secondbatch_normalized/moving_meanOGenerator/second/Generator/secondbatch_normalized/moving_mean/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@Generator/second/Generator/secondbatch_normalized/moving_mean*
validate_shape(*
_output_shapes	
:ђ
Ё
BGenerator/second/Generator/secondbatch_normalized/moving_mean/readIdentity=Generator/second/Generator/secondbatch_normalized/moving_mean*
T0*P
_classF
DBloc:@Generator/second/Generator/secondbatch_normalized/moving_mean*
_output_shapes	
:ђ
э
RGenerator/second/Generator/secondbatch_normalized/moving_variance/Initializer/onesConst*T
_classJ
HFloc:@Generator/second/Generator/secondbatch_normalized/moving_variance*
valueBђ*  ђ?*
dtype0*
_output_shapes	
:ђ
Ё
AGenerator/second/Generator/secondbatch_normalized/moving_variance
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *T
_classJ
HFloc:@Generator/second/Generator/secondbatch_normalized/moving_variance*
	container *
shape:ђ
ј
HGenerator/second/Generator/secondbatch_normalized/moving_variance/AssignAssignAGenerator/second/Generator/secondbatch_normalized/moving_varianceRGenerator/second/Generator/secondbatch_normalized/moving_variance/Initializer/ones*
T0*T
_classJ
HFloc:@Generator/second/Generator/secondbatch_normalized/moving_variance*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
Љ
FGenerator/second/Generator/secondbatch_normalized/moving_variance/readIdentityAGenerator/second/Generator/secondbatch_normalized/moving_variance*
_output_shapes	
:ђ*
T0*T
_classJ
HFloc:@Generator/second/Generator/secondbatch_normalized/moving_variance
є
AGenerator/second/Generator/secondbatch_normalized/batchnorm/add/yConst*
valueB
 *oЃ:*
dtype0*
_output_shapes
: 
э
?Generator/second/Generator/secondbatch_normalized/batchnorm/addAddFGenerator/second/Generator/secondbatch_normalized/moving_variance/readAGenerator/second/Generator/secondbatch_normalized/batchnorm/add/y*
_output_shapes	
:ђ*
T0
▒
AGenerator/second/Generator/secondbatch_normalized/batchnorm/RsqrtRsqrt?Generator/second/Generator/secondbatch_normalized/batchnorm/add*
T0*
_output_shapes	
:ђ
ь
?Generator/second/Generator/secondbatch_normalized/batchnorm/mulMulAGenerator/second/Generator/secondbatch_normalized/batchnorm/Rsqrt<Generator/second/Generator/secondbatch_normalized/gamma/read*
T0*
_output_shapes	
:ђ
Ш
AGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_1Mul8Generator/second/Generator/secondfully_connected/BiasAdd?Generator/second/Generator/secondbatch_normalized/batchnorm/mul*(
_output_shapes
:         ђ*
T0
з
AGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_2MulBGenerator/second/Generator/secondbatch_normalized/moving_mean/read?Generator/second/Generator/secondbatch_normalized/batchnorm/mul*
T0*
_output_shapes	
:ђ
В
?Generator/second/Generator/secondbatch_normalized/batchnorm/subSub;Generator/second/Generator/secondbatch_normalized/beta/readAGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_2*
_output_shapes	
:ђ*
T0
 
AGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1AddAGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_1?Generator/second/Generator/secondbatch_normalized/batchnorm/sub*(
_output_shapes
:         ђ*
T0
v
1Generator/second/Generator/secondleaky_relu/alphaConst*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
▀
/Generator/second/Generator/secondleaky_relu/mulMul1Generator/second/Generator/secondleaky_relu/alphaAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:         ђ
П
+Generator/second/Generator/secondleaky_reluMaximum/Generator/second/Generator/secondleaky_relu/mulAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*(
_output_shapes
:         ђ*
T0
ы
VGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/shapeConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
с
TGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/minConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB
 *зхй*
dtype0*
_output_shapes
: 
с
TGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/maxConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB
 *зх=*
dtype0*
_output_shapes
: 
Р
^Generator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformVGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/shape*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
seed2B*
dtype0* 
_output_shapes
:
ђђ*

seed
Ы
TGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/subSubTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/maxTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel
є
TGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/mulMul^Generator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/RandomUniformTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/sub*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel* 
_output_shapes
:
ђђ
Э
PGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniformAddTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/mulTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel* 
_output_shapes
:
ђђ
э
5Generator/third/Generator/thirdfully_connected/kernel
VariableV2*
dtype0* 
_output_shapes
:
ђђ*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
	container *
shape:
ђђ
ь
<Generator/third/Generator/thirdfully_connected/kernel/AssignAssign5Generator/third/Generator/thirdfully_connected/kernelPGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
validate_shape(* 
_output_shapes
:
ђђ
Ы
:Generator/third/Generator/thirdfully_connected/kernel/readIdentity5Generator/third/Generator/thirdfully_connected/kernel*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel* 
_output_shapes
:
ђђ
▄
EGenerator/third/Generator/thirdfully_connected/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:ђ*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
valueBђ*    
ж
3Generator/third/Generator/thirdfully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
	container *
shape:ђ
О
:Generator/third/Generator/thirdfully_connected/bias/AssignAssign3Generator/third/Generator/thirdfully_connected/biasEGenerator/third/Generator/thirdfully_connected/bias/Initializer/zeros*
_output_shapes	
:ђ*
use_locking(*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
validate_shape(
у
8Generator/third/Generator/thirdfully_connected/bias/readIdentity3Generator/third/Generator/thirdfully_connected/bias*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
_output_shapes	
:ђ
Ђ
5Generator/third/Generator/thirdfully_connected/MatMulMatMul+Generator/second/Generator/secondleaky_relu:Generator/third/Generator/thirdfully_connected/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b( 
Ч
6Generator/third/Generator/thirdfully_connected/BiasAddBiasAdd5Generator/third/Generator/thirdfully_connected/MatMul8Generator/third/Generator/thirdfully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:         ђ*
T0
▀
FGenerator/third/Generator/thirdbatch_normalized/gamma/Initializer/onesConst*
_output_shapes	
:ђ*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
valueBђ*  ђ?*
dtype0
ь
5Generator/third/Generator/thirdbatch_normalized/gamma
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
	container *
shape:ђ
я
<Generator/third/Generator/thirdbatch_normalized/gamma/AssignAssign5Generator/third/Generator/thirdbatch_normalized/gammaFGenerator/third/Generator/thirdbatch_normalized/gamma/Initializer/ones*
_output_shapes	
:ђ*
use_locking(*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
validate_shape(
ь
:Generator/third/Generator/thirdbatch_normalized/gamma/readIdentity5Generator/third/Generator/thirdbatch_normalized/gamma*
_output_shapes	
:ђ*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma
я
FGenerator/third/Generator/thirdbatch_normalized/beta/Initializer/zerosConst*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
в
4Generator/third/Generator/thirdbatch_normalized/beta
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
	container *
shape:ђ
█
;Generator/third/Generator/thirdbatch_normalized/beta/AssignAssign4Generator/third/Generator/thirdbatch_normalized/betaFGenerator/third/Generator/thirdbatch_normalized/beta/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
validate_shape(*
_output_shapes	
:ђ
Ж
9Generator/third/Generator/thirdbatch_normalized/beta/readIdentity4Generator/third/Generator/thirdbatch_normalized/beta*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
_output_shapes	
:ђ
В
MGenerator/third/Generator/thirdbatch_normalized/moving_mean/Initializer/zerosConst*N
_classD
B@loc:@Generator/third/Generator/thirdbatch_normalized/moving_mean*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
щ
;Generator/third/Generator/thirdbatch_normalized/moving_mean
VariableV2*N
_classD
B@loc:@Generator/third/Generator/thirdbatch_normalized/moving_mean*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name 
э
BGenerator/third/Generator/thirdbatch_normalized/moving_mean/AssignAssign;Generator/third/Generator/thirdbatch_normalized/moving_meanMGenerator/third/Generator/thirdbatch_normalized/moving_mean/Initializer/zeros*
T0*N
_classD
B@loc:@Generator/third/Generator/thirdbatch_normalized/moving_mean*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
 
@Generator/third/Generator/thirdbatch_normalized/moving_mean/readIdentity;Generator/third/Generator/thirdbatch_normalized/moving_mean*
_output_shapes	
:ђ*
T0*N
_classD
B@loc:@Generator/third/Generator/thirdbatch_normalized/moving_mean
з
PGenerator/third/Generator/thirdbatch_normalized/moving_variance/Initializer/onesConst*
_output_shapes	
:ђ*R
_classH
FDloc:@Generator/third/Generator/thirdbatch_normalized/moving_variance*
valueBђ*  ђ?*
dtype0
Ђ
?Generator/third/Generator/thirdbatch_normalized/moving_variance
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *R
_classH
FDloc:@Generator/third/Generator/thirdbatch_normalized/moving_variance*
	container *
shape:ђ
є
FGenerator/third/Generator/thirdbatch_normalized/moving_variance/AssignAssign?Generator/third/Generator/thirdbatch_normalized/moving_variancePGenerator/third/Generator/thirdbatch_normalized/moving_variance/Initializer/ones*
_output_shapes	
:ђ*
use_locking(*
T0*R
_classH
FDloc:@Generator/third/Generator/thirdbatch_normalized/moving_variance*
validate_shape(
І
DGenerator/third/Generator/thirdbatch_normalized/moving_variance/readIdentity?Generator/third/Generator/thirdbatch_normalized/moving_variance*
T0*R
_classH
FDloc:@Generator/third/Generator/thirdbatch_normalized/moving_variance*
_output_shapes	
:ђ
ё
?Generator/third/Generator/thirdbatch_normalized/batchnorm/add/yConst*
valueB
 *oЃ:*
dtype0*
_output_shapes
: 
ы
=Generator/third/Generator/thirdbatch_normalized/batchnorm/addAddDGenerator/third/Generator/thirdbatch_normalized/moving_variance/read?Generator/third/Generator/thirdbatch_normalized/batchnorm/add/y*
T0*
_output_shapes	
:ђ
Г
?Generator/third/Generator/thirdbatch_normalized/batchnorm/RsqrtRsqrt=Generator/third/Generator/thirdbatch_normalized/batchnorm/add*
T0*
_output_shapes	
:ђ
у
=Generator/third/Generator/thirdbatch_normalized/batchnorm/mulMul?Generator/third/Generator/thirdbatch_normalized/batchnorm/Rsqrt:Generator/third/Generator/thirdbatch_normalized/gamma/read*
T0*
_output_shapes	
:ђ
­
?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1Mul6Generator/third/Generator/thirdfully_connected/BiasAdd=Generator/third/Generator/thirdbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:         ђ
ь
?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2Mul@Generator/third/Generator/thirdbatch_normalized/moving_mean/read=Generator/third/Generator/thirdbatch_normalized/batchnorm/mul*
T0*
_output_shapes	
:ђ
Т
=Generator/third/Generator/thirdbatch_normalized/batchnorm/subSub9Generator/third/Generator/thirdbatch_normalized/beta/read?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2*
_output_shapes	
:ђ*
T0
щ
?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1Add?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1=Generator/third/Generator/thirdbatch_normalized/batchnorm/sub*(
_output_shapes
:         ђ*
T0
t
/Generator/third/Generator/thirdleaky_relu/alphaConst*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
┘
-Generator/third/Generator/thirdleaky_relu/mulMul/Generator/third/Generator/thirdleaky_relu/alpha?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*(
_output_shapes
:         ђ*
T0
О
)Generator/third/Generator/thirdleaky_reluMaximum-Generator/third/Generator/thirdleaky_relu/mul?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*(
_output_shapes
:         ђ*
T0
ы
VGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/shapeConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
с
TGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/minConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB
 *  ђй*
dtype0*
_output_shapes
: 
с
TGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB
 *  ђ=
Р
^Generator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformVGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/shape*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
seed2m*
dtype0* 
_output_shapes
:
ђђ*

seed
Ы
TGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/subSubTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/maxTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel
є
TGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/mulMul^Generator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/RandomUniformTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/sub*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:
ђђ
Э
PGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniformAddTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/mulTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:
ђђ
э
5Generator/forth/Generator/forthfully_connected/kernel
VariableV2*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
	container *
shape:
ђђ*
dtype0* 
_output_shapes
:
ђђ*
shared_name 
ь
<Generator/forth/Generator/forthfully_connected/kernel/AssignAssign5Generator/forth/Generator/forthfully_connected/kernelPGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
validate_shape(* 
_output_shapes
:
ђђ
Ы
:Generator/forth/Generator/forthfully_connected/kernel/readIdentity5Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:
ђђ*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel
У
UGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB:ђ
п
KGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros/ConstConst*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
ы
EGenerator/forth/Generator/forthfully_connected/bias/Initializer/zerosFillUGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros/shape_as_tensorKGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros/Const*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*

index_type0*
_output_shapes	
:ђ
ж
3Generator/forth/Generator/forthfully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
	container *
shape:ђ
О
:Generator/forth/Generator/forthfully_connected/bias/AssignAssign3Generator/forth/Generator/forthfully_connected/biasEGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias
у
8Generator/forth/Generator/forthfully_connected/bias/readIdentity3Generator/forth/Generator/forthfully_connected/bias*
_output_shapes	
:ђ*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias
 
5Generator/forth/Generator/forthfully_connected/MatMulMatMul)Generator/third/Generator/thirdleaky_relu:Generator/forth/Generator/forthfully_connected/kernel/read*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b( *
T0
Ч
6Generator/forth/Generator/forthfully_connected/BiasAddBiasAdd5Generator/forth/Generator/forthfully_connected/MatMul8Generator/forth/Generator/forthfully_connected/bias/read*(
_output_shapes
:         ђ*
T0*
data_formatNHWC
в
VGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones/shape_as_tensorConst*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB:ђ*
dtype0*
_output_shapes
:
█
LGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones/ConstConst*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
Ш
FGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/onesFillVGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones/shape_as_tensorLGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones/Const*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*

index_type0*
_output_shapes	
:ђ
ь
5Generator/forth/Generator/forthbatch_normalized/gamma
VariableV2*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ
я
<Generator/forth/Generator/forthbatch_normalized/gamma/AssignAssign5Generator/forth/Generator/forthbatch_normalized/gammaFGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:ђ
ь
:Generator/forth/Generator/forthbatch_normalized/gamma/readIdentity5Generator/forth/Generator/forthbatch_normalized/gamma*
_output_shapes	
:ђ*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma
Ж
VGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros/shape_as_tensorConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB:ђ*
dtype0*
_output_shapes
:
┌
LGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros/ConstConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
ш
FGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zerosFillVGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros/shape_as_tensorLGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros/Const*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*

index_type0*
_output_shapes	
:ђ
в
4Generator/forth/Generator/forthbatch_normalized/beta
VariableV2*
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
	container 
█
;Generator/forth/Generator/forthbatch_normalized/beta/AssignAssign4Generator/forth/Generator/forthbatch_normalized/betaFGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0
Ж
9Generator/forth/Generator/forthbatch_normalized/beta/readIdentity4Generator/forth/Generator/forthbatch_normalized/beta*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
_output_shapes	
:ђ
Э
]Generator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros/shape_as_tensorConst*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
valueB:ђ*
dtype0*
_output_shapes
:
У
SGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros/ConstConst*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
Љ
MGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zerosFill]Generator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros/shape_as_tensorSGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros/Const*
T0*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*

index_type0*
_output_shapes	
:ђ
щ
;Generator/forth/Generator/forthbatch_normalized/moving_mean
VariableV2*
shared_name *N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ
э
BGenerator/forth/Generator/forthbatch_normalized/moving_mean/AssignAssign;Generator/forth/Generator/forthbatch_normalized/moving_meanMGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros*
use_locking(*
T0*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
validate_shape(*
_output_shapes	
:ђ
 
@Generator/forth/Generator/forthbatch_normalized/moving_mean/readIdentity;Generator/forth/Generator/forthbatch_normalized/moving_mean*
T0*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
_output_shapes	
:ђ
 
`Generator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones/shape_as_tensorConst*
_output_shapes
:*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
valueB:ђ*
dtype0
№
VGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones/ConstConst*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
ъ
PGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/onesFill`Generator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones/shape_as_tensorVGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones/Const*
T0*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*

index_type0*
_output_shapes	
:ђ
Ђ
?Generator/forth/Generator/forthbatch_normalized/moving_variance
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
	container *
shape:ђ
є
FGenerator/forth/Generator/forthbatch_normalized/moving_variance/AssignAssign?Generator/forth/Generator/forthbatch_normalized/moving_variancePGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones*
_output_shapes	
:ђ*
use_locking(*
T0*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
validate_shape(
І
DGenerator/forth/Generator/forthbatch_normalized/moving_variance/readIdentity?Generator/forth/Generator/forthbatch_normalized/moving_variance*
T0*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
_output_shapes	
:ђ
ё
?Generator/forth/Generator/forthbatch_normalized/batchnorm/add/yConst*
valueB
 *oЃ:*
dtype0*
_output_shapes
: 
ы
=Generator/forth/Generator/forthbatch_normalized/batchnorm/addAddDGenerator/forth/Generator/forthbatch_normalized/moving_variance/read?Generator/forth/Generator/forthbatch_normalized/batchnorm/add/y*
T0*
_output_shapes	
:ђ
Г
?Generator/forth/Generator/forthbatch_normalized/batchnorm/RsqrtRsqrt=Generator/forth/Generator/forthbatch_normalized/batchnorm/add*
_output_shapes	
:ђ*
T0
у
=Generator/forth/Generator/forthbatch_normalized/batchnorm/mulMul?Generator/forth/Generator/forthbatch_normalized/batchnorm/Rsqrt:Generator/forth/Generator/forthbatch_normalized/gamma/read*
_output_shapes	
:ђ*
T0
­
?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1Mul6Generator/forth/Generator/forthfully_connected/BiasAdd=Generator/forth/Generator/forthbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:         ђ
ь
?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2Mul@Generator/forth/Generator/forthbatch_normalized/moving_mean/read=Generator/forth/Generator/forthbatch_normalized/batchnorm/mul*
_output_shapes	
:ђ*
T0
Т
=Generator/forth/Generator/forthbatch_normalized/batchnorm/subSub9Generator/forth/Generator/forthbatch_normalized/beta/read?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2*
T0*
_output_shapes	
:ђ
щ
?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1Add?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1=Generator/forth/Generator/forthbatch_normalized/batchnorm/sub*(
_output_shapes
:         ђ*
T0
t
/Generator/forth/Generator/forthleaky_relu/alphaConst*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
┘
-Generator/forth/Generator/forthleaky_relu/mulMul/Generator/forth/Generator/forthleaky_relu/alpha?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:         ђ
О
)Generator/forth/Generator/forthleaky_reluMaximum-Generator/forth/Generator/forthleaky_relu/mul?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*(
_output_shapes
:         ђ*
T0
│
7Generator/dense/kernel/Initializer/random_uniform/shapeConst*)
_class
loc:@Generator/dense/kernel*
valueB"     *
dtype0*
_output_shapes
:
Ц
5Generator/dense/kernel/Initializer/random_uniform/minConst*)
_class
loc:@Generator/dense/kernel*
valueB
 *zшkй*
dtype0*
_output_shapes
: 
Ц
5Generator/dense/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *)
_class
loc:@Generator/dense/kernel*
valueB
 *zшk=*
dtype0
є
?Generator/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform7Generator/dense/kernel/Initializer/random_uniform/shape*)
_class
loc:@Generator/dense/kernel*
seed2б*
dtype0* 
_output_shapes
:
ђљ*

seed*
T0
Ш
5Generator/dense/kernel/Initializer/random_uniform/subSub5Generator/dense/kernel/Initializer/random_uniform/max5Generator/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@Generator/dense/kernel*
_output_shapes
: 
і
5Generator/dense/kernel/Initializer/random_uniform/mulMul?Generator/dense/kernel/Initializer/random_uniform/RandomUniform5Generator/dense/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@Generator/dense/kernel* 
_output_shapes
:
ђљ
Ч
1Generator/dense/kernel/Initializer/random_uniformAdd5Generator/dense/kernel/Initializer/random_uniform/mul5Generator/dense/kernel/Initializer/random_uniform/min* 
_output_shapes
:
ђљ*
T0*)
_class
loc:@Generator/dense/kernel
╣
Generator/dense/kernel
VariableV2*
	container *
shape:
ђљ*
dtype0* 
_output_shapes
:
ђљ*
shared_name *)
_class
loc:@Generator/dense/kernel
ы
Generator/dense/kernel/AssignAssignGenerator/dense/kernel1Generator/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*)
_class
loc:@Generator/dense/kernel*
validate_shape(* 
_output_shapes
:
ђљ
Ћ
Generator/dense/kernel/readIdentityGenerator/dense/kernel*
T0*)
_class
loc:@Generator/dense/kernel* 
_output_shapes
:
ђљ
ъ
&Generator/dense/bias/Initializer/zerosConst*'
_class
loc:@Generator/dense/bias*
valueBљ*    *
dtype0*
_output_shapes	
:љ
Ф
Generator/dense/bias
VariableV2*
_output_shapes	
:љ*
shared_name *'
_class
loc:@Generator/dense/bias*
	container *
shape:љ*
dtype0
█
Generator/dense/bias/AssignAssignGenerator/dense/bias&Generator/dense/bias/Initializer/zeros*
_output_shapes	
:љ*
use_locking(*
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(
і
Generator/dense/bias/readIdentityGenerator/dense/bias*
_output_shapes	
:љ*
T0*'
_class
loc:@Generator/dense/bias
┴
Generator/dense/MatMulMatMul)Generator/forth/Generator/forthleaky_reluGenerator/dense/kernel/read*(
_output_shapes
:         љ*
transpose_a( *
transpose_b( *
T0
Ъ
Generator/dense/BiasAddBiasAddGenerator/dense/MatMulGenerator/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         љ
b
Generator/TanhTanhGenerator/dense/BiasAdd*(
_output_shapes
:         љ*
T0
w
Discriminator/realPlaceholder*
dtype0*(
_output_shapes
:         љ*
shape:         љ
Ђ
^Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/shapeConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
з
\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/minConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB
 *HYІй*
dtype0*
_output_shapes
: 
з
\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/maxConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB
 *HYІ=*
dtype0*
_output_shapes
: 
ч
fDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniform^Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
љђ*

seed*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
seed2┤*
dtype0
њ
\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/subSub\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/max\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel
д
\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/mulMulfDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniform\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/sub*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel* 
_output_shapes
:
љђ
ў
XDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniformAdd\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/mul\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/min*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel* 
_output_shapes
:
љђ*
T0
Є
=Discriminator/first/Discriminator/firstfully_connected/kernel
VariableV2*
	container *
shape:
љђ*
dtype0* 
_output_shapes
:
љђ*
shared_name *P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel
Ї
DDiscriminator/first/Discriminator/firstfully_connected/kernel/AssignAssign=Discriminator/first/Discriminator/firstfully_connected/kernelXDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
validate_shape(* 
_output_shapes
:
љђ
і
BDiscriminator/first/Discriminator/firstfully_connected/kernel/readIdentity=Discriminator/first/Discriminator/firstfully_connected/kernel*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel* 
_output_shapes
:
љђ
В
MDiscriminator/first/Discriminator/firstfully_connected/bias/Initializer/zerosConst*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
щ
;Discriminator/first/Discriminator/firstfully_connected/bias
VariableV2*
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
	container 
э
BDiscriminator/first/Discriminator/firstfully_connected/bias/AssignAssign;Discriminator/first/Discriminator/firstfully_connected/biasMDiscriminator/first/Discriminator/firstfully_connected/bias/Initializer/zeros*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
 
@Discriminator/first/Discriminator/firstfully_connected/bias/readIdentity;Discriminator/first/Discriminator/firstfully_connected/bias*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes	
:ђ*
T0
Э
=Discriminator/first/Discriminator/firstfully_connected/MatMulMatMulDiscriminator/realBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:         ђ*
transpose_a( 
ћ
>Discriminator/first/Discriminator/firstfully_connected/BiasAddBiasAdd=Discriminator/first/Discriminator/firstfully_connected/MatMul@Discriminator/first/Discriminator/firstfully_connected/bias/read*(
_output_shapes
:         ђ*
T0*
data_formatNHWC
|
7Discriminator/first/Discriminator/firstleaky_relu/alphaConst*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
У
5Discriminator/first/Discriminator/firstleaky_relu/mulMul7Discriminator/first/Discriminator/firstleaky_relu/alpha>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
Т
1Discriminator/first/Discriminator/firstleaky_reluMaximum5Discriminator/first/Discriminator/firstleaky_relu/mul>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*(
_output_shapes
:         ђ*
T0
Ё
`Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/shapeConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
э
^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/minConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB
 *зхй*
dtype0*
_output_shapes
: 
э
^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/maxConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB
 *зх=*
dtype0*
_output_shapes
: 
Ђ
hDiscriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniform`Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
ђђ*

seed*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
seed2К
џ
^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/subSub^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/max^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel
«
^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/mulMulhDiscriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/RandomUniform^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
ђђ*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel
а
ZDiscriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniformAdd^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/mul^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/min*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel* 
_output_shapes
:
ђђ
І
?Discriminator/second/Discriminator/secondfully_connected/kernel
VariableV2*
dtype0* 
_output_shapes
:
ђђ*
shared_name *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
	container *
shape:
ђђ
Ћ
FDiscriminator/second/Discriminator/secondfully_connected/kernel/AssignAssign?Discriminator/second/Discriminator/secondfully_connected/kernelZDiscriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(*
T0
љ
DDiscriminator/second/Discriminator/secondfully_connected/kernel/readIdentity?Discriminator/second/Discriminator/secondfully_connected/kernel*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel* 
_output_shapes
:
ђђ
­
ODiscriminator/second/Discriminator/secondfully_connected/bias/Initializer/zerosConst*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
§
=Discriminator/second/Discriminator/secondfully_connected/bias
VariableV2*
shared_name *P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ
 
DDiscriminator/second/Discriminator/secondfully_connected/bias/AssignAssign=Discriminator/second/Discriminator/secondfully_connected/biasODiscriminator/second/Discriminator/secondfully_connected/bias/Initializer/zeros*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0
Ё
BDiscriminator/second/Discriminator/secondfully_connected/bias/readIdentity=Discriminator/second/Discriminator/secondfully_connected/bias*
_output_shapes	
:ђ*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias
Џ
?Discriminator/second/Discriminator/secondfully_connected/MatMulMatMul1Discriminator/first/Discriminator/firstleaky_reluDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:         ђ*
transpose_a( 
џ
@Discriminator/second/Discriminator/secondfully_connected/BiasAddBiasAdd?Discriminator/second/Discriminator/secondfully_connected/MatMulBDiscriminator/second/Discriminator/secondfully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:         ђ*
T0
~
9Discriminator/second/Discriminator/secondleaky_relu/alphaConst*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
Ь
7Discriminator/second/Discriminator/secondleaky_relu/mulMul9Discriminator/second/Discriminator/secondleaky_relu/alpha@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
В
3Discriminator/second/Discriminator/secondleaky_reluMaximum7Discriminator/second/Discriminator/secondleaky_relu/mul@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
и
9Discriminator/out/kernel/Initializer/random_uniform/shapeConst*+
_class!
loc:@Discriminator/out/kernel*
valueB"      *
dtype0*
_output_shapes
:
Е
7Discriminator/out/kernel/Initializer/random_uniform/minConst*+
_class!
loc:@Discriminator/out/kernel*
valueB
 *IvЙ*
dtype0*
_output_shapes
: 
Е
7Discriminator/out/kernel/Initializer/random_uniform/maxConst*+
_class!
loc:@Discriminator/out/kernel*
valueB
 *Iv>*
dtype0*
_output_shapes
: 
І
ADiscriminator/out/kernel/Initializer/random_uniform/RandomUniformRandomUniform9Discriminator/out/kernel/Initializer/random_uniform/shape*
T0*+
_class!
loc:@Discriminator/out/kernel*
seed2┌*
dtype0*
_output_shapes
:	ђ*

seed
■
7Discriminator/out/kernel/Initializer/random_uniform/subSub7Discriminator/out/kernel/Initializer/random_uniform/max7Discriminator/out/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*+
_class!
loc:@Discriminator/out/kernel
Љ
7Discriminator/out/kernel/Initializer/random_uniform/mulMulADiscriminator/out/kernel/Initializer/random_uniform/RandomUniform7Discriminator/out/kernel/Initializer/random_uniform/sub*
_output_shapes
:	ђ*
T0*+
_class!
loc:@Discriminator/out/kernel
Ѓ
3Discriminator/out/kernel/Initializer/random_uniformAdd7Discriminator/out/kernel/Initializer/random_uniform/mul7Discriminator/out/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
:	ђ
╗
Discriminator/out/kernel
VariableV2*
shared_name *+
_class!
loc:@Discriminator/out/kernel*
	container *
shape:	ђ*
dtype0*
_output_shapes
:	ђ
Э
Discriminator/out/kernel/AssignAssignDiscriminator/out/kernel3Discriminator/out/kernel/Initializer/random_uniform*
use_locking(*
T0*+
_class!
loc:@Discriminator/out/kernel*
validate_shape(*
_output_shapes
:	ђ
џ
Discriminator/out/kernel/readIdentityDiscriminator/out/kernel*
T0*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
:	ђ
а
(Discriminator/out/bias/Initializer/zerosConst*)
_class
loc:@Discriminator/out/bias*
valueB*    *
dtype0*
_output_shapes
:
Г
Discriminator/out/bias
VariableV2*
shared_name *)
_class
loc:@Discriminator/out/bias*
	container *
shape:*
dtype0*
_output_shapes
:
Р
Discriminator/out/bias/AssignAssignDiscriminator/out/bias(Discriminator/out/bias/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@Discriminator/out/bias*
validate_shape(*
_output_shapes
:
Ј
Discriminator/out/bias/readIdentityDiscriminator/out/bias*
_output_shapes
:*
T0*)
_class
loc:@Discriminator/out/bias
╬
Discriminator/out/MatMulMatMul3Discriminator/second/Discriminator/secondleaky_reluDiscriminator/out/kernel/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
ц
Discriminator/out/BiasAddBiasAddDiscriminator/out/MatMulDiscriminator/out/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
q
Discriminator/out/SigmoidSigmoidDiscriminator/out/BiasAdd*'
_output_shapes
:         *
T0
Ш
?Discriminator/first_1/Discriminator/firstfully_connected/MatMulMatMulGenerator/TanhBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b( 
ў
@Discriminator/first_1/Discriminator/firstfully_connected/BiasAddBiasAdd?Discriminator/first_1/Discriminator/firstfully_connected/MatMul@Discriminator/first/Discriminator/firstfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         ђ
~
9Discriminator/first_1/Discriminator/firstleaky_relu/alphaConst*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
Ь
7Discriminator/first_1/Discriminator/firstleaky_relu/mulMul9Discriminator/first_1/Discriminator/firstleaky_relu/alpha@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
В
3Discriminator/first_1/Discriminator/firstleaky_reluMaximum7Discriminator/first_1/Discriminator/firstleaky_relu/mul@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*(
_output_shapes
:         ђ*
T0
Ъ
ADiscriminator/second_1/Discriminator/secondfully_connected/MatMulMatMul3Discriminator/first_1/Discriminator/firstleaky_reluDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b( 
ъ
BDiscriminator/second_1/Discriminator/secondfully_connected/BiasAddBiasAddADiscriminator/second_1/Discriminator/secondfully_connected/MatMulBDiscriminator/second/Discriminator/secondfully_connected/bias/read*(
_output_shapes
:         ђ*
T0*
data_formatNHWC
ђ
;Discriminator/second_1/Discriminator/secondleaky_relu/alphaConst*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
З
9Discriminator/second_1/Discriminator/secondleaky_relu/mulMul;Discriminator/second_1/Discriminator/secondleaky_relu/alphaBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
Ы
5Discriminator/second_1/Discriminator/secondleaky_reluMaximum9Discriminator/second_1/Discriminator/secondleaky_relu/mulBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*(
_output_shapes
:         ђ*
T0
м
Discriminator/out_1/MatMulMatMul5Discriminator/second_1/Discriminator/secondleaky_reluDiscriminator/out/kernel/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
е
Discriminator/out_1/BiasAddBiasAddDiscriminator/out_1/MatMulDiscriminator/out/bias/read*
data_formatNHWC*'
_output_shapes
:         *
T0
u
Discriminator/out_1/SigmoidSigmoidDiscriminator/out_1/BiasAdd*'
_output_shapes
:         *
T0
W
LogLogDiscriminator/out/Sigmoid*
T0*'
_output_shapes
:         
J
sub/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
`
subSubsub/xDiscriminator/out_1/Sigmoid*'
_output_shapes
:         *
T0
C
Log_1Logsub*
T0*'
_output_shapes
:         
H
addAddLogLog_1*
T0*'
_output_shapes
:         
V
ConstConst*
valueB"       *
dtype0*
_output_shapes
:
V
MeanMeanaddConst*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
1
NegNegMean*
_output_shapes
: *
T0
i
discriminator_loss/tagConst*
dtype0*
_output_shapes
: *#
valueB Bdiscriminator_loss
d
discriminator_lossHistogramSummarydiscriminator_loss/tagNeg*
T0*
_output_shapes
: 
L
sub_1/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
d
sub_1Subsub_1/xDiscriminator/out_1/Sigmoid*'
_output_shapes
:         *
T0
E
Log_2Logsub_1*
T0*'
_output_shapes
:         
X
Const_1Const*
_output_shapes
:*
valueB"       *
dtype0
\
Mean_1MeanLog_2Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
a
generator_loss/tagConst*
valueB Bgenerator_loss*
dtype0*
_output_shapes
: 
_
generator_lossHistogramSummarygenerator_loss/tagMean_1*
_output_shapes
: *
T0
R
gradients/ShapeConst*
_output_shapes
: *
valueB *
dtype0
X
gradients/grad_ys_0Const*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
N
gradients/Neg_grad/NegNeggradients/Fill*
_output_shapes
: *
T0
r
!gradients/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
ў
gradients/Mean_grad/ReshapeReshapegradients/Neg_grad/Neg!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
\
gradients/Mean_grad/ShapeShapeadd*
out_type0*
_output_shapes
:*
T0
ю
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*'
_output_shapes
:         *

Tmultiples0
^
gradients/Mean_grad/Shape_1Shapeadd*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
ќ
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
џ
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
ѓ
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0
ђ
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
ї
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*'
_output_shapes
:         
[
gradients/add_grad/ShapeShapeLog*
T0*
out_type0*
_output_shapes
:
_
gradients/add_grad/Shape_1ShapeLog_1*
out_type0*
_output_shapes
:*
T0
┤
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:         :         *
T0
ц
gradients/add_grad/SumSumgradients/Mean_grad/truediv(gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ќ
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
е
gradients/add_grad/Sum_1Sumgradients/Mean_grad/truediv*gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ю
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
┌
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*'
_output_shapes
:         *
T0*-
_class#
!loc:@gradients/add_grad/Reshape
Я
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*'
_output_shapes
:         
д
gradients/Log_grad/Reciprocal
ReciprocalDiscriminator/out/Sigmoid,^gradients/add_grad/tuple/control_dependency*'
_output_shapes
:         *
T0
Џ
gradients/Log_grad/mulMul+gradients/add_grad/tuple/control_dependencygradients/Log_grad/Reciprocal*'
_output_shapes
:         *
T0
ћ
gradients/Log_1_grad/Reciprocal
Reciprocalsub.^gradients/add_grad/tuple/control_dependency_1*'
_output_shapes
:         *
T0
А
gradients/Log_1_grad/mulMul-gradients/add_grad/tuple/control_dependency_1gradients/Log_1_grad/Reciprocal*'
_output_shapes
:         *
T0
е
4gradients/Discriminator/out/Sigmoid_grad/SigmoidGradSigmoidGradDiscriminator/out/Sigmoidgradients/Log_grad/mul*
T0*'
_output_shapes
:         
[
gradients/sub_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
u
gradients/sub_grad/Shape_1ShapeDiscriminator/out_1/Sigmoid*
_output_shapes
:*
T0*
out_type0
┤
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:         :         
А
gradients/sub_grad/SumSumgradients/Log_1_grad/mul(gradients/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
є
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
Ц
gradients/sub_grad/Sum_1Sumgradients/Log_1_grad/mul*gradients/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
T0*
_output_shapes
:
Џ
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
╔
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*
_output_shapes
: 
Я
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*'
_output_shapes
:         *
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1
х
4gradients/Discriminator/out/BiasAdd_grad/BiasAddGradBiasAddGrad4gradients/Discriminator/out/Sigmoid_grad/SigmoidGrad*
T0*
data_formatNHWC*
_output_shapes
:
»
9gradients/Discriminator/out/BiasAdd_grad/tuple/group_depsNoOp5^gradients/Discriminator/out/BiasAdd_grad/BiasAddGrad5^gradients/Discriminator/out/Sigmoid_grad/SigmoidGrad
║
Agradients/Discriminator/out/BiasAdd_grad/tuple/control_dependencyIdentity4gradients/Discriminator/out/Sigmoid_grad/SigmoidGrad:^gradients/Discriminator/out/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:         *
T0*G
_class=
;9loc:@gradients/Discriminator/out/Sigmoid_grad/SigmoidGrad
»
Cgradients/Discriminator/out/BiasAdd_grad/tuple/control_dependency_1Identity4gradients/Discriminator/out/BiasAdd_grad/BiasAddGrad:^gradients/Discriminator/out/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/Discriminator/out/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
├
6gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGradSigmoidGradDiscriminator/out_1/Sigmoid-gradients/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         
з
.gradients/Discriminator/out/MatMul_grad/MatMulMatMulAgradients/Discriminator/out/BiasAdd_grad/tuple/control_dependencyDiscriminator/out/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b(
ѓ
0gradients/Discriminator/out/MatMul_grad/MatMul_1MatMul3Discriminator/second/Discriminator/secondleaky_reluAgradients/Discriminator/out/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	ђ*
transpose_a(*
transpose_b( *
T0
ц
8gradients/Discriminator/out/MatMul_grad/tuple/group_depsNoOp/^gradients/Discriminator/out/MatMul_grad/MatMul1^gradients/Discriminator/out/MatMul_grad/MatMul_1
Г
@gradients/Discriminator/out/MatMul_grad/tuple/control_dependencyIdentity.gradients/Discriminator/out/MatMul_grad/MatMul9^gradients/Discriminator/out/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/Discriminator/out/MatMul_grad/MatMul*(
_output_shapes
:         ђ
ф
Bgradients/Discriminator/out/MatMul_grad/tuple/control_dependency_1Identity0gradients/Discriminator/out/MatMul_grad/MatMul_19^gradients/Discriminator/out/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/Discriminator/out/MatMul_grad/MatMul_1*
_output_shapes
:	ђ
╣
6gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGrad*
_output_shapes
:*
T0*
data_formatNHWC
х
;gradients/Discriminator/out_1/BiasAdd_grad/tuple/group_depsNoOp7^gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGrad7^gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGrad
┬
Cgradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGrad<^gradients/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGrad*'
_output_shapes
:         
и
Egradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency_1Identity6gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGrad<^gradients/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*I
_class?
=;loc:@gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGrad
┐
Hgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ShapeShape7Discriminator/second/Discriminator/secondleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
╩
Jgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_1Shape@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
╩
Jgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_2Shape@gradients/Discriminator/out/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
Њ
Ngradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
▒
Hgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zerosFillJgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_2Ngradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:         ђ
Ї
Ogradients/Discriminator/second/Discriminator/secondleaky_relu_grad/GreaterEqualGreaterEqual7Discriminator/second/Discriminator/secondleaky_relu/mul@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
─
Xgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ShapeJgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:         :         
с
Igradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SelectSelectOgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/GreaterEqual@gradients/Discriminator/out/MatMul_grad/tuple/control_dependencyHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zeros*
T0*(
_output_shapes
:         ђ
т
Kgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Select_1SelectOgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/GreaterEqualHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zeros@gradients/Discriminator/out/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:         ђ
▓
Fgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SumSumIgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SelectXgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
е
Jgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ReshapeReshapeFgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SumHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape*(
_output_shapes
:         ђ*
T0*
Tshape0
И
Hgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Sum_1SumKgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Select_1Zgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
«
Lgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1ReshapeHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Sum_1Jgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:         ђ
э
Sgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/group_depsNoOpK^gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ReshapeM^gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1
Џ
[gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependencyIdentityJgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ReshapeT^gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape*(
_output_shapes
:         ђ
А
]gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1IdentityLgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1T^gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ
э
0gradients/Discriminator/out_1/MatMul_grad/MatMulMatMulCgradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/out/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b(
ѕ
2gradients/Discriminator/out_1/MatMul_grad/MatMul_1MatMul5Discriminator/second_1/Discriminator/secondleaky_reluCgradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	ђ*
transpose_a(*
transpose_b( *
T0
ф
:gradients/Discriminator/out_1/MatMul_grad/tuple/group_depsNoOp1^gradients/Discriminator/out_1/MatMul_grad/MatMul3^gradients/Discriminator/out_1/MatMul_grad/MatMul_1
х
Bgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependencyIdentity0gradients/Discriminator/out_1/MatMul_grad/MatMul;^gradients/Discriminator/out_1/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/Discriminator/out_1/MatMul_grad/MatMul*(
_output_shapes
:         ђ
▓
Dgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency_1Identity2gradients/Discriminator/out_1/MatMul_grad/MatMul_1;^gradients/Discriminator/out_1/MatMul_grad/tuple/group_deps*E
_class;
97loc:@gradients/Discriminator/out_1/MatMul_grad/MatMul_1*
_output_shapes
:	ђ*
T0
Ў
gradients/AddNAddNCgradients/Discriminator/out/BiasAdd_grad/tuple/control_dependency_1Egradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency_1*
T0*G
_class=
;9loc:@gradients/Discriminator/out/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:
Ј
Lgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
╬
Ngradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape_1Shape@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
л
\gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ShapeNgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Б
Jgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/MulMul[gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
╗
Jgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/SumSumJgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Mul\gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
б
Ngradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ReshapeReshapeJgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/SumLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ъ
Lgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Mul_1Mul9Discriminator/second/Discriminator/secondleaky_relu/alpha[gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:         ђ
┴
Lgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Sum_1SumLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Mul_1^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
║
Pgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape_1ReshapeLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Sum_1Ngradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:         ђ
Ѓ
Wgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/group_depsNoOpO^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ReshapeQ^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape_1
Ў
_gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependencyIdentityNgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ReshapeX^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape*
_output_shapes
: 
▒
agradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1IdentityPgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape_1X^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:         ђ
├
Jgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeShape9Discriminator/second_1/Discriminator/secondleaky_relu/mul*
_output_shapes
:*
T0*
out_type0
╬
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1ShapeBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
╬
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_2ShapeBgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
Ћ
Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
и
Jgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zerosFillLgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_2Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros/Const*(
_output_shapes
:         ђ*
T0*

index_type0
Њ
Qgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualGreaterEqual9Discriminator/second_1/Discriminator/secondleaky_relu/mulBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
╩
Zgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeLgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:         :         
в
Kgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SelectSelectQgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualBgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependencyJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros*
T0*(
_output_shapes
:         ђ
ь
Mgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1SelectQgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zerosBgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency*(
_output_shapes
:         ђ*
T0
И
Hgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SumSumKgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SelectZgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
«
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeReshapeHgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SumJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         ђ
Й
Jgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1SumMgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1\gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
┤
Ngradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1ReshapeJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:         ђ
§
Ugradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_depsNoOpM^gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeO^gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1
Б
]gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependencyIdentityLgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeV^gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape*(
_output_shapes
:         ђ
Е
_gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1IdentityNgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1V^gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ
џ
gradients/AddN_1AddNBgradients/Discriminator/out/MatMul_grad/tuple/control_dependency_1Dgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/Discriminator/out/MatMul_grad/MatMul_1*
N*
_output_shapes
:	ђ
э
gradients/AddN_2AddN]gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1agradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1*
T0*_
_classU
SQloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:         ђ
╣
[gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_2*
data_formatNHWC*
_output_shapes	
:ђ*
T0
┘
`gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_2\^gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad
§
hgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_2a^gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*_
_classU
SQloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ*
T0
╠
jgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity[gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrada^gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*n
_classd
b`loc:@gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:ђ
Љ
Ngradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
м
Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1ShapeBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
о
^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ShapePgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Е
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/MulMul]gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependencyBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
┴
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumSumLgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
е
Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeReshapeLgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumNgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
ц
Ngradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul_1Mul;Discriminator/second_1/Discriminator/secondleaky_relu/alpha]gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency*(
_output_shapes
:         ђ*
T0
К
Ngradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1SumNgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul_1`gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
└
Rgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1ReshapeNgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:         ђ
Ѕ
Ygradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_depsNoOpQ^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeS^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1
А
agradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependencyIdentityPgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeZ^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*c
_classY
WUloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape*
_output_shapes
: *
T0
╣
cgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1IdentityRgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1Z^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:         ђ
У
Ugradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMulMatMulhgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b(*
T0
¤
Wgradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1MatMul1Discriminator/first/Discriminator/firstleaky_reluhgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
ђђ*
transpose_a(*
transpose_b( 
Ў
_gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/group_depsNoOpV^gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMulX^gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1
╔
ggradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyIdentityUgradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul`^gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul*(
_output_shapes
:         ђ
К
igradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1IdentityWgradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1`^gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
ђђ*
T0*j
_class`
^\loc:@gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1
§
gradients/AddN_3AddN_gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1cgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:         ђ*
T0*a
_classW
USloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1
╗
]gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_3*
T0*
data_formatNHWC*
_output_shapes	
:ђ
П
bgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_3^^gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad
Ѓ
jgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_3c^gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ
н
lgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity]gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGradc^gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*p
_classf
dbloc:@gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:ђ
╗
Fgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ShapeShape5Discriminator/first/Discriminator/firstleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
к
Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_1Shape>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
№
Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_2Shapeggradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
Љ
Lgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ф
Fgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zerosFillHgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_2Lgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:         ђ
Є
Mgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/GreaterEqualGreaterEqual5Discriminator/first/Discriminator/firstleaky_relu/mul>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*(
_output_shapes
:         ђ*
T0
Й
Vgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ShapeHgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_1*2
_output_shapes 
:         :         *
T0
ё
Ggradients/Discriminator/first/Discriminator/firstleaky_relu_grad/SelectSelectMgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/GreaterEqualggradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zeros*(
_output_shapes
:         ђ*
T0
є
Igradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Select_1SelectMgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/GreaterEqualFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zerosggradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:         ђ
г
Dgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/SumSumGgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/SelectVgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
б
Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ReshapeReshapeDgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/SumFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         ђ
▓
Fgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Sum_1SumIgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Select_1Xgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
е
Jgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1ReshapeFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Sum_1Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_1*(
_output_shapes
:         ђ*
T0*
Tshape0
ы
Qgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/group_depsNoOpI^gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ReshapeK^gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1
Њ
Ygradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependencyIdentityHgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ReshapeR^gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape*(
_output_shapes
:         ђ
Ў
[gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1IdentityJgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1R^gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ
В
Wgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMulMatMuljgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b(*
T0
Н
Ygradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1MatMul3Discriminator/first_1/Discriminator/firstleaky_relujgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
ђђ*
transpose_a(
Ъ
agradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_depsNoOpX^gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMulZ^gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1
Л
igradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyIdentityWgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMulb^gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:         ђ*
T0*j
_class`
^\loc:@gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul
¤
kgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1IdentityYgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1b^gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*l
_classb
`^loc:@gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
ђђ*
T0
Љ
gradients/AddN_4AddNjgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1lgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*n
_classd
b`loc:@gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:ђ
Ї
Jgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
╩
Lgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape_1Shape>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
╩
Zgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ShapeLgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Ю
Hgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/MulMulYgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependency>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
х
Hgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/SumSumHgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/MulZgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ю
Lgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ReshapeReshapeHgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/SumJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ў
Jgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Mul_1Mul7Discriminator/first/Discriminator/firstleaky_relu/alphaYgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependency*(
_output_shapes
:         ђ*
T0
╗
Jgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Sum_1SumJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Mul_1\gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
┤
Ngradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1ReshapeJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Sum_1Lgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:         ђ
§
Ugradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/group_depsNoOpM^gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ReshapeO^gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1
Љ
]gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependencyIdentityLgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ReshapeV^gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*_
_classU
SQloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape
Е
_gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityNgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1V^gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*a
_classW
USloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:         ђ*
T0
┐
Hgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ShapeShape7Discriminator/first_1/Discriminator/firstleaky_relu/mul*
out_type0*
_output_shapes
:*
T0
╩
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1Shape@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
з
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_2Shapeigradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
Њ
Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
▒
Hgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zerosFillJgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_2Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:         ђ
Ї
Ogradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualGreaterEqual7Discriminator/first_1/Discriminator/firstleaky_relu/mul@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
─
Xgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ShapeJgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:         :         
ї
Igradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SelectSelectOgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualigradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros*
T0*(
_output_shapes
:         ђ
ј
Kgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1SelectOgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zerosigradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:         ђ*
T0
▓
Fgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumSumIgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SelectXgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
е
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeReshapeFgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape*
Tshape0*(
_output_shapes
:         ђ*
T0
И
Hgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Sum_1SumKgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1Zgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
«
Lgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1ReshapeHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Sum_1Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:         ђ
э
Sgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_depsNoOpK^gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeM^gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1
Џ
[gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependencyIdentityJgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeT^gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape*(
_output_shapes
:         ђ
А
]gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1IdentityLgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1T^gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ
љ
gradients/AddN_5AddNigradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1kgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1*
T0*j
_class`
^\loc:@gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1*
N* 
_output_shapes
:
ђђ
ы
gradients/AddN_6AddN[gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1_gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1*
T0*]
_classS
QOloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:         ђ
и
Ygradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_6*
T0*
data_formatNHWC*
_output_shapes	
:ђ
Н
^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_6Z^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad
э
fgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_6_^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:         ђ*
T0*]
_classS
QOloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1
─
hgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityYgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad_^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:ђ*
T0*l
_classb
`^loc:@gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad
Ј
Lgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
╬
Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1Shape@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
л
\gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ShapeNgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Б
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/MulMul[gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
╗
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumSumJgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul\gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
б
Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeReshapeJgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumLgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ъ
Lgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul_1Mul9Discriminator/first_1/Discriminator/firstleaky_relu/alpha[gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:         ђ
┴
Lgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Sum_1SumLgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul_1^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
║
Pgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1ReshapeLgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Sum_1Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1*(
_output_shapes
:         ђ*
T0*
Tshape0
Ѓ
Wgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_depsNoOpO^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeQ^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1
Ў
_gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependencyIdentityNgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeX^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*a
_classW
USloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape
▒
agradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityPgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1X^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:         ђ
Р
Sgradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMulMatMulfgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
T0*(
_output_shapes
:         љ*
transpose_a( *
transpose_b(
г
Ugradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1MatMulDiscriminator/realfgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
љђ*
transpose_a(*
transpose_b( 
Њ
]gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/group_depsNoOpT^gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMulV^gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
┴
egradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentitySgradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul^^gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:         љ*
T0*f
_class\
ZXloc:@gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul
┐
ggradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityUgradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1^^gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
љђ*
T0*h
_class^
\Zloc:@gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
э
gradients/AddN_7AddN]gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1agradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1*_
_classU
SQloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:         ђ*
T0
╣
[gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_7*
T0*
data_formatNHWC*
_output_shapes	
:ђ
┘
`gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_7\^gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad
§
hgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_7a^gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ
╠
jgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity[gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrada^gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*n
_classd
b`loc:@gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:ђ*
T0
Т
Ugradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulMatMulhgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
T0*(
_output_shapes
:         љ*
transpose_a( *
transpose_b(
г
Wgradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1MatMulGenerator/Tanhhgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
љђ*
transpose_a(*
transpose_b( *
T0
Ў
_gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_depsNoOpV^gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulX^gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
╔
ggradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentityUgradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul`^gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*h
_class^
\Zloc:@gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul*(
_output_shapes
:         љ*
T0
К
igradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityWgradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1`^gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
љђ*
T0*j
_class`
^\loc:@gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
І
gradients/AddN_8AddNhgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1jgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1*l
_classb
`^loc:@gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:ђ*
T0
і
gradients/AddN_9AddNggradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1igradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1*
N* 
_output_shapes
:
љђ*
T0*h
_class^
\Zloc:@gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
«
beta1_power/initial_valueConst*
_output_shapes
: *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueB
 *fff?*
dtype0
┐
beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
	container *
shape: 
я
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
џ
beta1_power/readIdentitybeta1_power*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes
: 
«
beta2_power/initial_valueConst*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueB
 *wЙ?*
dtype0*
_output_shapes
: 
┐
beta2_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias
я
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
џ
beta2_power/readIdentitybeta2_power*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes
: 
Є
dDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
ы
ZDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/ConstConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Г
TDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zerosFilldDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorZDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
љђ*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*

index_type0
ї
BDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
љђ*
shared_name *P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
	container *
shape:
љђ
Њ
IDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/AssignAssignBDiscriminator/first/Discriminator/firstfully_connected/kernel/AdamTDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
validate_shape(* 
_output_shapes
:
љђ
ћ
GDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/readIdentityBDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel* 
_output_shapes
:
љђ
Ѕ
fDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
з
\Discriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
│
VDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zerosFillfDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensor\Discriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*

index_type0* 
_output_shapes
:
љђ
ј
DDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1
VariableV2*
shape:
љђ*
dtype0* 
_output_shapes
:
љђ*
shared_name *P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
	container 
Ў
KDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/AssignAssignDDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1VDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
љђ*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel
ў
IDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/readIdentityDDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel* 
_output_shapes
:
љђ
ы
RDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/Initializer/zerosConst*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
■
@Discriminator/first/Discriminator/firstfully_connected/bias/Adam
VariableV2*
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ
є
GDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/AssignAssign@Discriminator/first/Discriminator/firstfully_connected/bias/AdamRDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:ђ
Ѕ
EDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/readIdentity@Discriminator/first/Discriminator/firstfully_connected/bias/Adam*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes	
:ђ
з
TDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:ђ*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueBђ*    
ђ
BDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1
VariableV2*
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
	container 
ї
IDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/AssignAssignBDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1TDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:ђ
Ї
GDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/readIdentityBDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes	
:ђ
І
fDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
ш
\Discriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB
 *    *
dtype0
х
VDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zerosFillfDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensor\Discriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros/Const*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*

index_type0* 
_output_shapes
:
ђђ
љ
DDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam
VariableV2* 
_output_shapes
:
ђђ*
shared_name *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
	container *
shape:
ђђ*
dtype0
Џ
KDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/AssignAssignDDiscriminator/second/Discriminator/secondfully_connected/kernel/AdamVDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel
џ
IDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/readIdentityDDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam* 
_output_shapes
:
ђђ*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel
Ї
hDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
э
^Discriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB
 *    
╗
XDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zerosFillhDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensor^Discriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros/Const*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*

index_type0* 
_output_shapes
:
ђђ*
T0
њ
FDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1
VariableV2*
	container *
shape:
ђђ*
dtype0* 
_output_shapes
:
ђђ*
shared_name *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel
А
MDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/AssignAssignFDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1XDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:
ђђ
ъ
KDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/readIdentityFDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1* 
_output_shapes
:
ђђ*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel
ш
TDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/Initializer/zerosConst*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
ѓ
BDiscriminator/second/Discriminator/secondfully_connected/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
	container *
shape:ђ
ј
IDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/AssignAssignBDiscriminator/second/Discriminator/secondfully_connected/bias/AdamTDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/Initializer/zeros*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0
Ј
GDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/readIdentityBDiscriminator/second/Discriminator/secondfully_connected/bias/Adam*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
_output_shapes	
:ђ
э
VDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/Initializer/zerosConst*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
ё
DDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
	container *
shape:ђ
ћ
KDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/AssignAssignDDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1VDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/Initializer/zeros*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0
Њ
IDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/readIdentityDDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
_output_shapes	
:ђ
│
/Discriminator/out/kernel/Adam/Initializer/zerosConst*+
_class!
loc:@Discriminator/out/kernel*
valueB	ђ*    *
dtype0*
_output_shapes
:	ђ
└
Discriminator/out/kernel/Adam
VariableV2*
	container *
shape:	ђ*
dtype0*
_output_shapes
:	ђ*
shared_name *+
_class!
loc:@Discriminator/out/kernel
■
$Discriminator/out/kernel/Adam/AssignAssignDiscriminator/out/kernel/Adam/Discriminator/out/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:	ђ*
use_locking(*
T0*+
_class!
loc:@Discriminator/out/kernel
ц
"Discriminator/out/kernel/Adam/readIdentityDiscriminator/out/kernel/Adam*
T0*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
:	ђ
х
1Discriminator/out/kernel/Adam_1/Initializer/zerosConst*+
_class!
loc:@Discriminator/out/kernel*
valueB	ђ*    *
dtype0*
_output_shapes
:	ђ
┬
Discriminator/out/kernel/Adam_1
VariableV2*+
_class!
loc:@Discriminator/out/kernel*
	container *
shape:	ђ*
dtype0*
_output_shapes
:	ђ*
shared_name 
ё
&Discriminator/out/kernel/Adam_1/AssignAssignDiscriminator/out/kernel/Adam_11Discriminator/out/kernel/Adam_1/Initializer/zeros*
_output_shapes
:	ђ*
use_locking(*
T0*+
_class!
loc:@Discriminator/out/kernel*
validate_shape(
е
$Discriminator/out/kernel/Adam_1/readIdentityDiscriminator/out/kernel/Adam_1*
T0*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
:	ђ
Ц
-Discriminator/out/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*)
_class
loc:@Discriminator/out/bias*
valueB*    
▓
Discriminator/out/bias/Adam
VariableV2*)
_class
loc:@Discriminator/out/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
ы
"Discriminator/out/bias/Adam/AssignAssignDiscriminator/out/bias/Adam-Discriminator/out/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*)
_class
loc:@Discriminator/out/bias
Ў
 Discriminator/out/bias/Adam/readIdentityDiscriminator/out/bias/Adam*)
_class
loc:@Discriminator/out/bias*
_output_shapes
:*
T0
Д
/Discriminator/out/bias/Adam_1/Initializer/zerosConst*)
_class
loc:@Discriminator/out/bias*
valueB*    *
dtype0*
_output_shapes
:
┤
Discriminator/out/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *)
_class
loc:@Discriminator/out/bias*
	container *
shape:
э
$Discriminator/out/bias/Adam_1/AssignAssignDiscriminator/out/bias/Adam_1/Discriminator/out/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@Discriminator/out/bias*
validate_shape(*
_output_shapes
:
Ю
"Discriminator/out/bias/Adam_1/readIdentityDiscriminator/out/bias/Adam_1*
T0*)
_class
loc:@Discriminator/out/bias*
_output_shapes
:
W
Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *иQ9
O

Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
O

Adam/beta2Const*
valueB
 *wЙ?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
й
SAdam/update_Discriminator/first/Discriminator/firstfully_connected/kernel/ApplyAdam	ApplyAdam=Discriminator/first/Discriminator/firstfully_connected/kernelBDiscriminator/first/Discriminator/firstfully_connected/kernel/AdamDDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_9* 
_output_shapes
:
љђ*
use_locking( *
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
use_nesterov( 
«
QAdam/update_Discriminator/first/Discriminator/firstfully_connected/bias/ApplyAdam	ApplyAdam;Discriminator/first/Discriminator/firstfully_connected/bias@Discriminator/first/Discriminator/firstfully_connected/bias/AdamBDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_8*
use_locking( *
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
use_nesterov( *
_output_shapes	
:ђ
К
UAdam/update_Discriminator/second/Discriminator/secondfully_connected/kernel/ApplyAdam	ApplyAdam?Discriminator/second/Discriminator/secondfully_connected/kernelDDiscriminator/second/Discriminator/secondfully_connected/kernel/AdamFDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_5*
use_locking( *
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
use_nesterov( * 
_output_shapes
:
ђђ
И
SAdam/update_Discriminator/second/Discriminator/secondfully_connected/bias/ApplyAdam	ApplyAdam=Discriminator/second/Discriminator/secondfully_connected/biasBDiscriminator/second/Discriminator/secondfully_connected/bias/AdamDDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_4*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
use_nesterov( *
_output_shapes	
:ђ*
use_locking( *
T0
Ѓ
.Adam/update_Discriminator/out/kernel/ApplyAdam	ApplyAdamDiscriminator/out/kernelDiscriminator/out/kernel/AdamDiscriminator/out/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_1*
T0*+
_class!
loc:@Discriminator/out/kernel*
use_nesterov( *
_output_shapes
:	ђ*
use_locking( 
Ы
,Adam/update_Discriminator/out/bias/ApplyAdam	ApplyAdamDiscriminator/out/biasDiscriminator/out/bias/AdamDiscriminator/out/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN*
use_locking( *
T0*)
_class
loc:@Discriminator/out/bias*
use_nesterov( *
_output_shapes
:
о
Adam/mulMulbeta1_power/read
Adam/beta1R^Adam/update_Discriminator/first/Discriminator/firstfully_connected/bias/ApplyAdamT^Adam/update_Discriminator/first/Discriminator/firstfully_connected/kernel/ApplyAdam-^Adam/update_Discriminator/out/bias/ApplyAdam/^Adam/update_Discriminator/out/kernel/ApplyAdamT^Adam/update_Discriminator/second/Discriminator/secondfully_connected/bias/ApplyAdamV^Adam/update_Discriminator/second/Discriminator/secondfully_connected/kernel/ApplyAdam*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes
: 
к
Adam/AssignAssignbeta1_powerAdam/mul*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes
: *
use_locking( 
п

Adam/mul_1Mulbeta2_power/read
Adam/beta2R^Adam/update_Discriminator/first/Discriminator/firstfully_connected/bias/ApplyAdamT^Adam/update_Discriminator/first/Discriminator/firstfully_connected/kernel/ApplyAdam-^Adam/update_Discriminator/out/bias/ApplyAdam/^Adam/update_Discriminator/out/kernel/ApplyAdamT^Adam/update_Discriminator/second/Discriminator/secondfully_connected/bias/ApplyAdamV^Adam/update_Discriminator/second/Discriminator/secondfully_connected/kernel/ApplyAdam*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes
: 
╩
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes
: 
Р
AdamNoOp^Adam/Assign^Adam/Assign_1R^Adam/update_Discriminator/first/Discriminator/firstfully_connected/bias/ApplyAdamT^Adam/update_Discriminator/first/Discriminator/firstfully_connected/kernel/ApplyAdam-^Adam/update_Discriminator/out/bias/ApplyAdam/^Adam/update_Discriminator/out/kernel/ApplyAdamT^Adam/update_Discriminator/second/Discriminator/secondfully_connected/bias/ApplyAdamV^Adam/update_Discriminator/second/Discriminator/secondfully_connected/kernel/ApplyAdam
T
gradients_1/ShapeConst*
_output_shapes
: *
valueB *
dtype0
Z
gradients_1/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ђ?
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
_output_shapes
: *
T0*

index_type0
v
%gradients_1/Mean_1_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
џ
gradients_1/Mean_1_grad/ReshapeReshapegradients_1/Fill%gradients_1/Mean_1_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
b
gradients_1/Mean_1_grad/ShapeShapeLog_2*
T0*
out_type0*
_output_shapes
:
е
gradients_1/Mean_1_grad/TileTilegradients_1/Mean_1_grad/Reshapegradients_1/Mean_1_grad/Shape*'
_output_shapes
:         *

Tmultiples0*
T0
d
gradients_1/Mean_1_grad/Shape_1ShapeLog_2*
_output_shapes
:*
T0*
out_type0
b
gradients_1/Mean_1_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
g
gradients_1/Mean_1_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
б
gradients_1/Mean_1_grad/ProdProdgradients_1/Mean_1_grad/Shape_1gradients_1/Mean_1_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
i
gradients_1/Mean_1_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
д
gradients_1/Mean_1_grad/Prod_1Prodgradients_1/Mean_1_grad/Shape_2gradients_1/Mean_1_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
c
!gradients_1/Mean_1_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
ј
gradients_1/Mean_1_grad/MaximumMaximumgradients_1/Mean_1_grad/Prod_1!gradients_1/Mean_1_grad/Maximum/y*
T0*
_output_shapes
: 
ї
 gradients_1/Mean_1_grad/floordivFloorDivgradients_1/Mean_1_grad/Prodgradients_1/Mean_1_grad/Maximum*
_output_shapes
: *
T0
є
gradients_1/Mean_1_grad/CastCast gradients_1/Mean_1_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
ў
gradients_1/Mean_1_grad/truedivRealDivgradients_1/Mean_1_grad/Tilegradients_1/Mean_1_grad/Cast*'
_output_shapes
:         *
T0
і
!gradients_1/Log_2_grad/Reciprocal
Reciprocalsub_1 ^gradients_1/Mean_1_grad/truediv*'
_output_shapes
:         *
T0
Ќ
gradients_1/Log_2_grad/mulMulgradients_1/Mean_1_grad/truediv!gradients_1/Log_2_grad/Reciprocal*'
_output_shapes
:         *
T0
_
gradients_1/sub_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
y
gradients_1/sub_1_grad/Shape_1ShapeDiscriminator/out_1/Sigmoid*
T0*
out_type0*
_output_shapes
:
└
,gradients_1/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_1_grad/Shapegradients_1/sub_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Ф
gradients_1/sub_1_grad/SumSumgradients_1/Log_2_grad/mul,gradients_1/sub_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
њ
gradients_1/sub_1_grad/ReshapeReshapegradients_1/sub_1_grad/Sumgradients_1/sub_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
»
gradients_1/sub_1_grad/Sum_1Sumgradients_1/Log_2_grad/mul.gradients_1/sub_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
b
gradients_1/sub_1_grad/NegNeggradients_1/sub_1_grad/Sum_1*
_output_shapes
:*
T0
Д
 gradients_1/sub_1_grad/Reshape_1Reshapegradients_1/sub_1_grad/Neggradients_1/sub_1_grad/Shape_1*'
_output_shapes
:         *
T0*
Tshape0
s
'gradients_1/sub_1_grad/tuple/group_depsNoOp^gradients_1/sub_1_grad/Reshape!^gradients_1/sub_1_grad/Reshape_1
┘
/gradients_1/sub_1_grad/tuple/control_dependencyIdentitygradients_1/sub_1_grad/Reshape(^gradients_1/sub_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/sub_1_grad/Reshape*
_output_shapes
: *
T0
­
1gradients_1/sub_1_grad/tuple/control_dependency_1Identity gradients_1/sub_1_grad/Reshape_1(^gradients_1/sub_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/sub_1_grad/Reshape_1*'
_output_shapes
:         
╔
8gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGradSigmoidGradDiscriminator/out_1/Sigmoid1gradients_1/sub_1_grad/tuple/control_dependency_1*'
_output_shapes
:         *
T0
й
8gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGradBiasAddGrad8gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad*
T0*
data_formatNHWC*
_output_shapes
:
╗
=gradients_1/Discriminator/out_1/BiasAdd_grad/tuple/group_depsNoOp9^gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGrad9^gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad
╩
Egradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyIdentity8gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad>^gradients_1/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad*'
_output_shapes
:         
┐
Ggradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency_1Identity8gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGrad>^gradients_1/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
ч
2gradients_1/Discriminator/out_1/MatMul_grad/MatMulMatMulEgradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/out/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b(
ї
4gradients_1/Discriminator/out_1/MatMul_grad/MatMul_1MatMul5Discriminator/second_1/Discriminator/secondleaky_reluEgradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	ђ*
transpose_a(*
transpose_b( *
T0
░
<gradients_1/Discriminator/out_1/MatMul_grad/tuple/group_depsNoOp3^gradients_1/Discriminator/out_1/MatMul_grad/MatMul5^gradients_1/Discriminator/out_1/MatMul_grad/MatMul_1
й
Dgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependencyIdentity2gradients_1/Discriminator/out_1/MatMul_grad/MatMul=^gradients_1/Discriminator/out_1/MatMul_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/Discriminator/out_1/MatMul_grad/MatMul*(
_output_shapes
:         ђ
║
Fgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependency_1Identity4gradients_1/Discriminator/out_1/MatMul_grad/MatMul_1=^gradients_1/Discriminator/out_1/MatMul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/Discriminator/out_1/MatMul_grad/MatMul_1*
_output_shapes
:	ђ
┼
Lgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeShape9Discriminator/second_1/Discriminator/secondleaky_relu/mul*
_output_shapes
:*
T0*
out_type0
л
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1ShapeBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
м
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_2ShapeDgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
Ќ
Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
й
Lgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zerosFillNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_2Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:         ђ
Ћ
Sgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualGreaterEqual9Discriminator/second_1/Discriminator/secondleaky_relu/mulBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*(
_output_shapes
:         ђ*
T0
л
\gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:         :         
з
Mgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SelectSelectSgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualDgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependencyLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros*
T0*(
_output_shapes
:         ђ
ш
Ogradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1SelectSgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zerosDgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:         ђ
Й
Jgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SumSumMgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select\gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
┤
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeReshapeJgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SumLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         ђ
─
Lgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1SumOgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
║
Pgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1ReshapeLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1*
Tshape0*(
_output_shapes
:         ђ*
T0
Ѓ
Wgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_depsNoOpO^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeQ^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1
Ф
_gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependencyIdentityNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeX^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_deps*a
_classW
USloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape*(
_output_shapes
:         ђ*
T0
▒
agradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1IdentityPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1X^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ
Њ
Pgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
н
Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1ShapeBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
▄
`gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ShapeRgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Г
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/MulMul_gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependencyBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*(
_output_shapes
:         ђ*
T0
К
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumSumNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul`gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
«
Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeReshapeNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
е
Pgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul_1Mul;Discriminator/second_1/Discriminator/secondleaky_relu/alpha_gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:         ђ
═
Pgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1SumPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul_1bgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
к
Tgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1ReshapePgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1*(
_output_shapes
:         ђ*
T0*
Tshape0
Ј
[gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_depsNoOpS^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeU^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1
Е
cgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependencyIdentityRgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape\^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape*
_output_shapes
: 
┴
egradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1IdentityTgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1\^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:         ђ*
T0*g
_class]
[Yloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1
Ѓ
gradients_1/AddNAddNagradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1egradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1*
T0*c
_classY
WUloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:         ђ
й
_gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN*
T0*
data_formatNHWC*
_output_shapes	
:ђ
р
dgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN`^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad
Ѕ
lgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddNe^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ
▄
ngradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity_gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrade^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*r
_classh
fdloc:@gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:ђ
­
Ygradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMulMatMullgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b(
┘
[gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1MatMul3Discriminator/first_1/Discriminator/firstleaky_relulgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
ђђ*
transpose_a(*
transpose_b( *
T0
Ц
cgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_depsNoOpZ^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul\^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1
┘
kgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyIdentityYgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMuld^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*l
_classb
`^loc:@gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul*(
_output_shapes
:         ђ
О
mgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1Identity[gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1d^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*n
_classd
b`loc:@gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
ђђ*
T0
┴
Jgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ShapeShape7Discriminator/first_1/Discriminator/firstleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
╠
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1Shape@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
э
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_2Shapekgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
Ћ
Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
и
Jgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zerosFillLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_2Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:         ђ
Ј
Qgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualGreaterEqual7Discriminator/first_1/Discriminator/firstleaky_relu/mul@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
╩
Zgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ShapeLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1*2
_output_shapes 
:         :         *
T0
ћ
Kgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SelectSelectQgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualkgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros*(
_output_shapes
:         ђ*
T0
ќ
Mgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1SelectQgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeroskgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:         ђ
И
Hgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumSumKgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SelectZgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
«
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeReshapeHgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         ђ
Й
Jgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Sum_1SumMgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1\gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
┤
Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1ReshapeJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Sum_1Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1*(
_output_shapes
:         ђ*
T0*
Tshape0
§
Ugradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_depsNoOpM^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeO^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1
Б
]gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependencyIdentityLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeV^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape*(
_output_shapes
:         ђ
Е
_gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1IdentityNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1V^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ
Љ
Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
л
Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1Shape@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
о
^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ShapePgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Д
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/MulMul]gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
┴
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumSumLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
е
Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeReshapeLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
б
Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul_1Mul9Discriminator/first_1/Discriminator/firstleaky_relu/alpha]gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:         ђ
К
Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Sum_1SumNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul_1`gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
└
Rgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1ReshapeNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Sum_1Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:         ђ
Ѕ
Ygradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_depsNoOpQ^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeS^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1
А
agradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependencyIdentityPgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeZ^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape*
_output_shapes
: 
╣
cgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityRgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1Z^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:         ђ
 
gradients_1/AddN_1AddN_gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1cgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1*
T0*a
_classW
USloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:         ђ
й
]gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_1*
T0*
data_formatNHWC*
_output_shapes	
:ђ
▀
bgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_1^^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad
Ё
jgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_1c^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*a
_classW
USloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ*
T0
н
lgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity]gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradc^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:ђ*
T0*p
_classf
dbloc:@gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad
Ж
Wgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulMatMuljgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
T0*(
_output_shapes
:         љ*
transpose_a( *
transpose_b(
░
Ygradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1MatMulGenerator/Tanhjgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
љђ*
transpose_a(*
transpose_b( *
T0
Ъ
agradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_depsNoOpX^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulZ^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
Л
igradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentityWgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulb^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*j
_class`
^\loc:@gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul*(
_output_shapes
:         љ
¤
kgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityYgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1b^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
љђ*
T0*l
_classb
`^loc:@gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
Р
(gradients_1/Generator/Tanh_grad/TanhGradTanhGradGenerator/Tanhigradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:         љ
ф
4gradients_1/Generator/dense/BiasAdd_grad/BiasAddGradBiasAddGrad(gradients_1/Generator/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes	
:љ
Б
9gradients_1/Generator/dense/BiasAdd_grad/tuple/group_depsNoOp)^gradients_1/Generator/Tanh_grad/TanhGrad5^gradients_1/Generator/dense/BiasAdd_grad/BiasAddGrad
Б
Agradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependencyIdentity(gradients_1/Generator/Tanh_grad/TanhGrad:^gradients_1/Generator/dense/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/Generator/Tanh_grad/TanhGrad*(
_output_shapes
:         љ
░
Cgradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependency_1Identity4gradients_1/Generator/dense/BiasAdd_grad/BiasAddGrad:^gradients_1/Generator/dense/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/Generator/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:љ
ы
.gradients_1/Generator/dense/MatMul_grad/MatMulMatMulAgradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependencyGenerator/dense/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b(
щ
0gradients_1/Generator/dense/MatMul_grad/MatMul_1MatMul)Generator/forth/Generator/forthleaky_reluAgradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
ђљ*
transpose_a(*
transpose_b( 
ц
8gradients_1/Generator/dense/MatMul_grad/tuple/group_depsNoOp/^gradients_1/Generator/dense/MatMul_grad/MatMul1^gradients_1/Generator/dense/MatMul_grad/MatMul_1
Г
@gradients_1/Generator/dense/MatMul_grad/tuple/control_dependencyIdentity.gradients_1/Generator/dense/MatMul_grad/MatMul9^gradients_1/Generator/dense/MatMul_grad/tuple/group_deps*(
_output_shapes
:         ђ*
T0*A
_class7
53loc:@gradients_1/Generator/dense/MatMul_grad/MatMul
Ф
Bgradients_1/Generator/dense/MatMul_grad/tuple/control_dependency_1Identity0gradients_1/Generator/dense/MatMul_grad/MatMul_19^gradients_1/Generator/dense/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/Generator/dense/MatMul_grad/MatMul_1* 
_output_shapes
:
ђљ
Г
@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/ShapeShape-Generator/forth/Generator/forthleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
┴
Bgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_1Shape?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
out_type0*
_output_shapes
:*
T0
┬
Bgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_2Shape@gradients_1/Generator/dense/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
І
Fgradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Ў
@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/zerosFillBgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_2Fgradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:         ђ
Щ
Ggradients_1/Generator/forth/Generator/forthleaky_relu_grad/GreaterEqualGreaterEqual-Generator/forth/Generator/forthleaky_relu/mul?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:         ђ
г
Pgradients_1/Generator/forth/Generator/forthleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/ShapeBgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╦
Agradients_1/Generator/forth/Generator/forthleaky_relu_grad/SelectSelectGgradients_1/Generator/forth/Generator/forthleaky_relu_grad/GreaterEqual@gradients_1/Generator/dense/MatMul_grad/tuple/control_dependency@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros*(
_output_shapes
:         ђ*
T0
═
Cgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Select_1SelectGgradients_1/Generator/forth/Generator/forthleaky_relu_grad/GreaterEqual@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros@gradients_1/Generator/dense/MatMul_grad/tuple/control_dependency*(
_output_shapes
:         ђ*
T0
џ
>gradients_1/Generator/forth/Generator/forthleaky_relu_grad/SumSumAgradients_1/Generator/forth/Generator/forthleaky_relu_grad/SelectPgradients_1/Generator/forth/Generator/forthleaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
љ
Bgradients_1/Generator/forth/Generator/forthleaky_relu_grad/ReshapeReshape>gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Sum@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         ђ
а
@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Sum_1SumCgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Select_1Rgradients_1/Generator/forth/Generator/forthleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ќ
Dgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1Reshape@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Sum_1Bgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_1*
Tshape0*(
_output_shapes
:         ђ*
T0
▀
Kgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/group_depsNoOpC^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/ReshapeE^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1
ч
Sgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependencyIdentityBgradients_1/Generator/forth/Generator/forthleaky_relu_grad/ReshapeL^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/group_deps*(
_output_shapes
:         ђ*
T0*U
_classK
IGloc:@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape
Ђ
Ugradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency_1IdentityDgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1L^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/group_deps*W
_classM
KIloc:@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ*
T0
Є
Dgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
┼
Fgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape_1Shape?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
И
Tgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ShapeFgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
њ
Bgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/MulMulSgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:         ђ
Б
Bgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/SumSumBgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/MulTgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
і
Fgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ReshapeReshapeBgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/SumDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
ё
Dgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Mul_1Mul/Generator/forth/Generator/forthleaky_relu/alphaSgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency*(
_output_shapes
:         ђ*
T0
Е
Dgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Sum_1SumDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Mul_1Vgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
б
Hgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape_1ReshapeDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Sum_1Fgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:         ђ
в
Ogradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/group_depsNoOpG^gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ReshapeI^gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape_1
щ
Wgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/control_dependencyIdentityFgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ReshapeP^gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape*
_output_shapes
: 
Љ
Ygradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/control_dependency_1IdentityHgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape_1P^gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/group_deps*[
_classQ
OMloc:@gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:         ђ*
T0
р
gradients_1/AddN_2AddNUgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency_1Ygradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/control_dependency_1*
T0*W
_classM
KIloc:@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:         ђ
Н
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/ShapeShape?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
Б
Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Shape_1Const*
valueB:ђ*
dtype0*
_output_shapes
:
Ь
fgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/ShapeXgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ќ
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/SumSumgradients_1/AddN_2fgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
м
Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/ReshapeReshapeTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/SumVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         ђ
Џ
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_2hgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
╦
Zgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1ReshapeVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Sum_1Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:ђ
А
agradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/group_depsNoOpY^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape[^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1
М
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependencyIdentityXgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshapeb^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*(
_output_shapes
:         ђ*
T0*k
_classa
_]loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape
╠
kgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1IdentityZgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1b^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:ђ
╠
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/ShapeShape6Generator/forth/Generator/forthfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Б
Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Shape_1Const*
valueB:ђ*
dtype0*
_output_shapes
:
Ь
fgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/ShapeXgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
И
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/MulMuligradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency=Generator/forth/Generator/forthbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:         ђ
┘
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/SumSumTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Mulfgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
м
Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/ReshapeReshapeTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/SumVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         ђ
│
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Mul_1Mul6Generator/forth/Generator/forthfully_connected/BiasAddigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:         ђ
▀
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Sum_1SumVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Mul_1hgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╦
Zgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1ReshapeVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Sum_1Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Shape_1*
Tshape0*
_output_shapes	
:ђ*
T0
А
agradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/group_depsNoOpY^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape[^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1
М
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyIdentityXgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshapeb^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:         ђ
╠
kgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityZgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1b^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*
_output_shapes	
:ђ*
T0*m
_classc
a_loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1
В
Rgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/NegNegkgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1*
_output_shapes	
:ђ*
T0
ф
_gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/group_depsNoOpl^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1S^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/Neg
О
ggradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependencyIdentitykgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1`^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:ђ
И
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1IdentityRgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/Neg`^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/Neg*
_output_shapes	
:ђ
і
Sgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:ђ
б
Xgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/group_depsNoOpj^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyT^gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/BiasAddGrad
м
`gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependencyIdentityigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyY^gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:         ђ
г
bgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/BiasAddGradY^gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:ђ
Ф
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/MulMuligradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1=Generator/forth/Generator/forthbatch_normalized/batchnorm/mul*
_output_shapes	
:ђ*
T0
░
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul_1Muligradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1@Generator/forth/Generator/forthbatch_normalized/moving_mean/read*
_output_shapes	
:ђ*
T0
Ў
agradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/group_depsNoOpU^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/MulW^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul_1
Й
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependencyIdentityTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mulb^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul*
_output_shapes	
:ђ
─
kgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul_1b^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:ђ*
T0*i
_class_
][loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul_1
╬
Mgradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMulMatMul`gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependency:Generator/forth/Generator/forthfully_connected/kernel/read*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b(*
T0
и
Ogradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul_1MatMul)Generator/third/Generator/thirdleaky_relu`gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
ђђ*
transpose_a(*
transpose_b( *
T0
Ђ
Wgradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/group_depsNoOpN^gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMulP^gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul_1
Е
_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependencyIdentityMgradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMulX^gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul*(
_output_shapes
:         ђ
Д
agradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul_1X^gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
ђђ
њ
gradients_1/AddN_3AddNkgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1kgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*m
_classc
a_loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:ђ
¤
Rgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/MulMulgradients_1/AddN_3:Generator/forth/Generator/forthbatch_normalized/gamma/read*
T0*
_output_shapes	
:ђ
о
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_3?Generator/forth/Generator/forthbatch_normalized/batchnorm/Rsqrt*
T0*
_output_shapes	
:ђ
Њ
_gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/group_depsNoOpS^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/MulU^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1
Х
ggradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/control_dependencyIdentityRgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul`^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul*
_output_shapes	
:ђ
╝
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1`^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1*
_output_shapes	
:ђ
Г
@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/ShapeShape-Generator/third/Generator/thirdleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
┴
Bgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_1Shape?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
р
Bgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_2Shape_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
І
Fgradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ў
@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/zerosFillBgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_2Fgradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros/Const*(
_output_shapes
:         ђ*
T0*

index_type0
Щ
Ggradients_1/Generator/third/Generator/thirdleaky_relu_grad/GreaterEqualGreaterEqual-Generator/third/Generator/thirdleaky_relu/mul?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*(
_output_shapes
:         ђ*
T0
г
Pgradients_1/Generator/third/Generator/thirdleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/ShapeBgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ж
Agradients_1/Generator/third/Generator/thirdleaky_relu_grad/SelectSelectGgradients_1/Generator/third/Generator/thirdleaky_relu_grad/GreaterEqual_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros*(
_output_shapes
:         ђ*
T0
В
Cgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Select_1SelectGgradients_1/Generator/third/Generator/thirdleaky_relu_grad/GreaterEqual@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:         ђ
џ
>gradients_1/Generator/third/Generator/thirdleaky_relu_grad/SumSumAgradients_1/Generator/third/Generator/thirdleaky_relu_grad/SelectPgradients_1/Generator/third/Generator/thirdleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
љ
Bgradients_1/Generator/third/Generator/thirdleaky_relu_grad/ReshapeReshape>gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Sum@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape*
Tshape0*(
_output_shapes
:         ђ*
T0
а
@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Sum_1SumCgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Select_1Rgradients_1/Generator/third/Generator/thirdleaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ќ
Dgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1Reshape@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Sum_1Bgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:         ђ
▀
Kgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/group_depsNoOpC^gradients_1/Generator/third/Generator/thirdleaky_relu_grad/ReshapeE^gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1
ч
Sgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependencyIdentityBgradients_1/Generator/third/Generator/thirdleaky_relu_grad/ReshapeL^gradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/group_deps*(
_output_shapes
:         ђ*
T0*U
_classK
IGloc:@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape
Ђ
Ugradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency_1IdentityDgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1L^gradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ
Є
Dgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
┼
Fgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Shape_1Shape?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
И
Tgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ShapeFgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
њ
Bgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/MulMulSgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:         ђ
Б
Bgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/SumSumBgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/MulTgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
і
Fgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ReshapeReshapeBgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/SumDgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ё
Dgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Mul_1Mul/Generator/third/Generator/thirdleaky_relu/alphaSgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency*(
_output_shapes
:         ђ*
T0
Е
Dgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Sum_1SumDgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Mul_1Vgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
б
Hgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1ReshapeDgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Sum_1Fgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:         ђ
в
Ogradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/group_depsNoOpG^gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ReshapeI^gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1
щ
Wgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/control_dependencyIdentityFgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ReshapeP^gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape*
_output_shapes
: 
Љ
Ygradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/control_dependency_1IdentityHgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1P^gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:         ђ
р
gradients_1/AddN_4AddNUgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency_1Ygradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/control_dependency_1*
T0*W
_classM
KIloc:@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:         ђ
Н
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/ShapeShape?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
Б
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape_1Const*
valueB:ђ*
dtype0*
_output_shapes
:
Ь
fgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/ShapeXgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ќ
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/SumSumgradients_1/AddN_4fgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
м
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/ReshapeReshapeTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/SumVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape*
Tshape0*(
_output_shapes
:         ђ*
T0
Џ
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_4hgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╦
Zgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1ReshapeVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Sum_1Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:ђ
А
agradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/group_depsNoOpY^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape[^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1
М
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependencyIdentityXgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshapeb^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape*(
_output_shapes
:         ђ
╠
kgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1IdentityZgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1b^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:ђ
╠
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/ShapeShape6Generator/third/Generator/thirdfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Б
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:ђ
Ь
fgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/ShapeXgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
И
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/MulMuligradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency=Generator/third/Generator/thirdbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:         ђ
┘
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/SumSumTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Mulfgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
м
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/ReshapeReshapeTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/SumVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         ђ
│
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Mul_1Mul6Generator/third/Generator/thirdfully_connected/BiasAddigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:         ђ
▀
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Sum_1SumVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Mul_1hgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╦
Zgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1ReshapeVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Sum_1Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:ђ
А
agradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/group_depsNoOpY^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape[^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1
М
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyIdentityXgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshapeb^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*k
_classa
_]loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:         ђ*
T0
╠
kgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityZgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1b^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:ђ
В
Rgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/NegNegkgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:ђ
ф
_gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/group_depsNoOpl^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1S^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/Neg
О
ggradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependencyIdentitykgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1`^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/group_deps*m
_classc
a_loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:ђ*
T0
И
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1IdentityRgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/Neg`^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/Neg*
_output_shapes	
:ђ
і
Sgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency*
data_formatNHWC*
_output_shapes	
:ђ*
T0
б
Xgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/group_depsNoOpj^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyT^gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/BiasAddGrad
м
`gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependencyIdentityigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyY^gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:         ђ
г
bgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/BiasAddGradY^gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:ђ
Ф
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/MulMuligradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1=Generator/third/Generator/thirdbatch_normalized/batchnorm/mul*
_output_shapes	
:ђ*
T0
░
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul_1Muligradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1@Generator/third/Generator/thirdbatch_normalized/moving_mean/read*
T0*
_output_shapes	
:ђ
Ў
agradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/group_depsNoOpU^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/MulW^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul_1
Й
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependencyIdentityTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mulb^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul*
_output_shapes	
:ђ
─
kgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul_1b^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:ђ*
T0*i
_class_
][loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul_1
╬
Mgradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMulMatMul`gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency:Generator/third/Generator/thirdfully_connected/kernel/read*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b(*
T0
╣
Ogradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul_1MatMul+Generator/second/Generator/secondleaky_relu`gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
ђђ*
transpose_a(*
transpose_b( 
Ђ
Wgradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/group_depsNoOpN^gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMulP^gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul_1
Е
_gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependencyIdentityMgradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMulX^gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul*(
_output_shapes
:         ђ
Д
agradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul_1X^gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
ђђ
њ
gradients_1/AddN_5AddNkgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1kgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1*
_output_shapes	
:ђ*
T0*m
_classc
a_loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
N
¤
Rgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/MulMulgradients_1/AddN_5:Generator/third/Generator/thirdbatch_normalized/gamma/read*
_output_shapes	
:ђ*
T0
о
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_5?Generator/third/Generator/thirdbatch_normalized/batchnorm/Rsqrt*
T0*
_output_shapes	
:ђ
Њ
_gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/group_depsNoOpS^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/MulU^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul_1
Х
ggradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/control_dependencyIdentityRgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul`^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul*
_output_shapes	
:ђ
╝
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul_1`^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul_1*
_output_shapes	
:ђ
▒
Bgradients_1/Generator/second/Generator/secondleaky_relu_grad/ShapeShape/Generator/second/Generator/secondleaky_relu/mul*
_output_shapes
:*
T0*
out_type0
┼
Dgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_1ShapeAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
с
Dgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_2Shape_gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependency*
out_type0*
_output_shapes
:*
T0
Ї
Hgradients_1/Generator/second/Generator/secondleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ъ
Bgradients_1/Generator/second/Generator/secondleaky_relu_grad/zerosFillDgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_2Hgradients_1/Generator/second/Generator/secondleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:         ђ
ђ
Igradients_1/Generator/second/Generator/secondleaky_relu_grad/GreaterEqualGreaterEqual/Generator/second/Generator/secondleaky_relu/mulAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:         ђ
▓
Rgradients_1/Generator/second/Generator/secondleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsBgradients_1/Generator/second/Generator/secondleaky_relu_grad/ShapeDgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_1*2
_output_shapes 
:         :         *
T0
­
Cgradients_1/Generator/second/Generator/secondleaky_relu_grad/SelectSelectIgradients_1/Generator/second/Generator/secondleaky_relu_grad/GreaterEqual_gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependencyBgradients_1/Generator/second/Generator/secondleaky_relu_grad/zeros*(
_output_shapes
:         ђ*
T0
Ы
Egradients_1/Generator/second/Generator/secondleaky_relu_grad/Select_1SelectIgradients_1/Generator/second/Generator/secondleaky_relu_grad/GreaterEqualBgradients_1/Generator/second/Generator/secondleaky_relu_grad/zeros_gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:         ђ
а
@gradients_1/Generator/second/Generator/secondleaky_relu_grad/SumSumCgradients_1/Generator/second/Generator/secondleaky_relu_grad/SelectRgradients_1/Generator/second/Generator/secondleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ќ
Dgradients_1/Generator/second/Generator/secondleaky_relu_grad/ReshapeReshape@gradients_1/Generator/second/Generator/secondleaky_relu_grad/SumBgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         ђ
д
Bgradients_1/Generator/second/Generator/secondleaky_relu_grad/Sum_1SumEgradients_1/Generator/second/Generator/secondleaky_relu_grad/Select_1Tgradients_1/Generator/second/Generator/secondleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ю
Fgradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1ReshapeBgradients_1/Generator/second/Generator/secondleaky_relu_grad/Sum_1Dgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:         ђ
т
Mgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/group_depsNoOpE^gradients_1/Generator/second/Generator/secondleaky_relu_grad/ReshapeG^gradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1
Ѓ
Ugradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependencyIdentityDgradients_1/Generator/second/Generator/secondleaky_relu_grad/ReshapeN^gradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/group_deps*W
_classM
KIloc:@gradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape*(
_output_shapes
:         ђ*
T0
Ѕ
Wgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependency_1IdentityFgradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1N^gradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ
Ѕ
Fgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
╔
Hgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape_1ShapeAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
Й
Vgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ShapeHgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
ў
Dgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/MulMulUgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependencyAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:         ђ
Е
Dgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/SumSumDgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/MulVgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
љ
Hgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ReshapeReshapeDgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/SumFgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
і
Fgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Mul_1Mul1Generator/second/Generator/secondleaky_relu/alphaUgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:         ђ
»
Fgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Sum_1SumFgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Mul_1Xgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
е
Jgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape_1ReshapeFgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Sum_1Hgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape_1*(
_output_shapes
:         ђ*
T0*
Tshape0
ы
Qgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/group_depsNoOpI^gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ReshapeK^gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape_1
Ђ
Ygradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/control_dependencyIdentityHgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ReshapeR^gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*[
_classQ
OMloc:@gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape
Ў
[gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/control_dependency_1IdentityJgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape_1R^gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/group_deps*]
_classS
QOloc:@gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:         ђ*
T0
у
gradients_1/AddN_6AddNWgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependency_1[gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/control_dependency_1*
T0*Y
_classO
MKloc:@gradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:         ђ
┘
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/ShapeShapeAGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_1*
_output_shapes
:*
T0*
out_type0
Ц
Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape_1Const*
valueB:ђ*
dtype0*
_output_shapes
:
З
hgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/ShapeZgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Џ
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/SumSumgradients_1/AddN_6hgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
п
Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/ReshapeReshapeVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/SumXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         ђ
Ъ
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_6jgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Л
\gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1ReshapeXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Sum_1Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:ђ
Д
cgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/group_depsNoOp[^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape]^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1
█
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependencyIdentityZgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshaped^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape*(
_output_shapes
:         ђ
н
mgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1Identity\gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1d^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*
T0*o
_classe
caloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:ђ
л
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/ShapeShape8Generator/second/Generator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ц
Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Shape_1Const*
_output_shapes
:*
valueB:ђ*
dtype0
З
hgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/ShapeZgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Й
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/MulMulkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency?Generator/second/Generator/secondbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:         ђ
▀
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/SumSumVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Mulhgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
п
Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/ReshapeReshapeVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/SumXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Shape*(
_output_shapes
:         ђ*
T0*
Tshape0
╣
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Mul_1Mul8Generator/second/Generator/secondfully_connected/BiasAddkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency*(
_output_shapes
:         ђ*
T0
т
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Sum_1SumXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Mul_1jgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Л
\gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1ReshapeXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Sum_1Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:ђ
Д
cgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/group_depsNoOp[^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape]^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1
█
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyIdentityZgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshaped^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:         ђ
н
mgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1Identity\gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1d^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*
T0*o
_classe
caloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:ђ
­
Tgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/NegNegmgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1*
_output_shapes	
:ђ*
T0
░
agradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/group_depsNoOpn^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1U^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/Neg
▀
igradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependencyIdentitymgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1b^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
T0*o
_classe
caloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:ђ
└
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/Negb^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/Neg*
_output_shapes	
:ђ
ј
Ugradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:ђ
е
Zgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/group_depsNoOpl^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyV^gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/BiasAddGrad
┌
bgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitykgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency[^gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/group_deps*m
_classc
a_loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:         ђ*
T0
┤
dgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityUgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/BiasAddGrad[^gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:ђ
▒
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/MulMulkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1?Generator/second/Generator/secondbatch_normalized/batchnorm/mul*
_output_shapes	
:ђ*
T0
Х
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul_1Mulkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1BGenerator/second/Generator/secondbatch_normalized/moving_mean/read*
_output_shapes	
:ђ*
T0
Ъ
cgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/group_depsNoOpW^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/MulY^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul_1
к
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependencyIdentityVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Muld^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul*
_output_shapes	
:ђ
╠
mgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul_1d^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*k
_classa
_]loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:ђ*
T0
н
Ogradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMulMatMulbgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependency<Generator/second/Generator/secondfully_connected/kernel/read*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b(*
T0
╗
Qgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul_1MatMul)Generator/first/Generator/firstleaky_relubgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
ђђ*
transpose_a(*
transpose_b( *
T0
Є
Ygradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/group_depsNoOpP^gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMulR^gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul_1
▒
agradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependencyIdentityOgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMulZ^gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul*(
_output_shapes
:         ђ
»
cgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency_1IdentityQgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul_1Z^gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
ђђ*
T0*d
_classZ
XVloc:@gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul_1
ў
gradients_1/AddN_7AddNmgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1mgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*o
_classe
caloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:ђ
М
Tgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/MulMulgradients_1/AddN_7<Generator/second/Generator/secondbatch_normalized/gamma/read*
T0*
_output_shapes	
:ђ
┌
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_7AGenerator/second/Generator/secondbatch_normalized/batchnorm/Rsqrt*
T0*
_output_shapes	
:ђ
Ў
agradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/group_depsNoOpU^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/MulW^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul_1
Й
igradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/control_dependencyIdentityTgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mulb^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
_output_shapes	
:ђ*
T0*g
_class]
[Yloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul
─
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1IdentityVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul_1b^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
_output_shapes	
:ђ*
T0*i
_class_
][loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul_1
Г
@gradients_1/Generator/first/Generator/firstleaky_relu_grad/ShapeShape-Generator/first/Generator/firstleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
И
Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_1Shape6Generator/first/Generator/firstfully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
с
Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_2Shapeagradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency*
out_type0*
_output_shapes
:*
T0
І
Fgradients_1/Generator/first/Generator/firstleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ў
@gradients_1/Generator/first/Generator/firstleaky_relu_grad/zerosFillBgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_2Fgradients_1/Generator/first/Generator/firstleaky_relu_grad/zeros/Const*(
_output_shapes
:         ђ*
T0*

index_type0
ы
Ggradients_1/Generator/first/Generator/firstleaky_relu_grad/GreaterEqualGreaterEqual-Generator/first/Generator/firstleaky_relu/mul6Generator/first/Generator/firstfully_connected/BiasAdd*(
_output_shapes
:         ђ*
T0
г
Pgradients_1/Generator/first/Generator/firstleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients_1/Generator/first/Generator/firstleaky_relu_grad/ShapeBgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_1*2
_output_shapes 
:         :         *
T0
В
Agradients_1/Generator/first/Generator/firstleaky_relu_grad/SelectSelectGgradients_1/Generator/first/Generator/firstleaky_relu_grad/GreaterEqualagradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency@gradients_1/Generator/first/Generator/firstleaky_relu_grad/zeros*
T0*(
_output_shapes
:         ђ
Ь
Cgradients_1/Generator/first/Generator/firstleaky_relu_grad/Select_1SelectGgradients_1/Generator/first/Generator/firstleaky_relu_grad/GreaterEqual@gradients_1/Generator/first/Generator/firstleaky_relu_grad/zerosagradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:         ђ
џ
>gradients_1/Generator/first/Generator/firstleaky_relu_grad/SumSumAgradients_1/Generator/first/Generator/firstleaky_relu_grad/SelectPgradients_1/Generator/first/Generator/firstleaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
љ
Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/ReshapeReshape>gradients_1/Generator/first/Generator/firstleaky_relu_grad/Sum@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         ђ
а
@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Sum_1SumCgradients_1/Generator/first/Generator/firstleaky_relu_grad/Select_1Rgradients_1/Generator/first/Generator/firstleaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ќ
Dgradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1Reshape@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Sum_1Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:         ђ
▀
Kgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/group_depsNoOpC^gradients_1/Generator/first/Generator/firstleaky_relu_grad/ReshapeE^gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1
ч
Sgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependencyIdentityBgradients_1/Generator/first/Generator/firstleaky_relu_grad/ReshapeL^gradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape*(
_output_shapes
:         ђ
Ђ
Ugradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependency_1IdentityDgradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1L^gradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ
Є
Dgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
╝
Fgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape_1Shape6Generator/first/Generator/firstfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
И
Tgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ShapeFgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ѕ
Bgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/MulMulSgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependency6Generator/first/Generator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
Б
Bgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/SumSumBgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/MulTgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
і
Fgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ReshapeReshapeBgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/SumDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
ё
Dgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Mul_1Mul/Generator/first/Generator/firstleaky_relu/alphaSgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependency*(
_output_shapes
:         ђ*
T0
Е
Dgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Sum_1SumDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Mul_1Vgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
б
Hgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape_1ReshapeDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Sum_1Fgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:         ђ
в
Ogradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/group_depsNoOpG^gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ReshapeI^gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape_1
щ
Wgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/control_dependencyIdentityFgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ReshapeP^gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*Y
_classO
MKloc:@gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape
Љ
Ygradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityHgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape_1P^gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:         ђ
р
gradients_1/AddN_8AddNUgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependency_1Ygradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/control_dependency_1*
T0*W
_classM
KIloc:@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:         ђ
│
Sgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_8*
T0*
data_formatNHWC*
_output_shapes	
:ђ
╦
Xgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_8T^gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/BiasAddGrad
у
`gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_8Y^gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ
г
bgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/BiasAddGradY^gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/group_deps*f
_class\
ZXloc:@gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:ђ*
T0
═
Mgradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMulMatMul`gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependency:Generator/first/Generator/firstfully_connected/kernel/read*'
_output_shapes
:         d*
transpose_a( *
transpose_b(*
T0
ю
Ogradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1MatMulGenerator/noise`gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	dђ*
transpose_a(*
transpose_b( 
Ђ
Wgradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/group_depsNoOpN^gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMulP^gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1
е
_gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentityMgradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMulX^gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/group_deps*`
_classV
TRloc:@gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul*'
_output_shapes
:         d*
T0
д
agradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1X^gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1*
_output_shapes
:	dђ
Ѕ
beta1_power_1/initial_valueConst*'
_class
loc:@Generator/dense/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
џ
beta1_power_1
VariableV2*
dtype0*
_output_shapes
: *
shared_name *'
_class
loc:@Generator/dense/bias*
	container *
shape: 
й
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
use_locking(*
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes
: 
w
beta1_power_1/readIdentitybeta1_power_1*'
_class
loc:@Generator/dense/bias*
_output_shapes
: *
T0
Ѕ
beta2_power_1/initial_valueConst*'
_class
loc:@Generator/dense/bias*
valueB
 *wЙ?*
dtype0*
_output_shapes
: 
џ
beta2_power_1
VariableV2*
dtype0*
_output_shapes
: *
shared_name *'
_class
loc:@Generator/dense/bias*
	container *
shape: 
й
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*'
_class
loc:@Generator/dense/bias
w
beta2_power_1/readIdentitybeta2_power_1*'
_class
loc:@Generator/dense/bias*
_output_shapes
: *
T0
э
\Generator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB"d   ђ   *
dtype0*
_output_shapes
:
р
RGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB
 *    *
dtype0
ї
LGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zerosFill\Generator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorRGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros/Const*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*

index_type0*
_output_shapes
:	dђ
Щ
:Generator/first/Generator/firstfully_connected/kernel/Adam
VariableV2*
dtype0*
_output_shapes
:	dђ*
shared_name *H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
	container *
shape:	dђ
Ы
AGenerator/first/Generator/firstfully_connected/kernel/Adam/AssignAssign:Generator/first/Generator/firstfully_connected/kernel/AdamLGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
validate_shape(*
_output_shapes
:	dђ
ч
?Generator/first/Generator/firstfully_connected/kernel/Adam/readIdentity:Generator/first/Generator/firstfully_connected/kernel/Adam*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
_output_shapes
:	dђ
щ
^Generator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB"d   ђ   
с
TGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
њ
NGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zerosFill^Generator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorTGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*

index_type0*
_output_shapes
:	dђ
Ч
<Generator/first/Generator/firstfully_connected/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes
:	dђ*
shared_name *H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
	container *
shape:	dђ
Э
CGenerator/first/Generator/firstfully_connected/kernel/Adam_1/AssignAssign<Generator/first/Generator/firstfully_connected/kernel/Adam_1NGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	dђ*
use_locking(*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel
 
AGenerator/first/Generator/firstfully_connected/kernel/Adam_1/readIdentity<Generator/first/Generator/firstfully_connected/kernel/Adam_1*
_output_shapes
:	dђ*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel
р
JGenerator/first/Generator/firstfully_connected/bias/Adam/Initializer/zerosConst*
_output_shapes	
:ђ*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
valueBђ*    *
dtype0
Ь
8Generator/first/Generator/firstfully_connected/bias/Adam
VariableV2*
shared_name *F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ
Т
?Generator/first/Generator/firstfully_connected/bias/Adam/AssignAssign8Generator/first/Generator/firstfully_connected/bias/AdamJGenerator/first/Generator/firstfully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:ђ
ы
=Generator/first/Generator/firstfully_connected/bias/Adam/readIdentity8Generator/first/Generator/firstfully_connected/bias/Adam*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
_output_shapes	
:ђ
с
LGenerator/first/Generator/firstfully_connected/bias/Adam_1/Initializer/zerosConst*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
­
:Generator/first/Generator/firstfully_connected/bias/Adam_1
VariableV2*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias
В
AGenerator/first/Generator/firstfully_connected/bias/Adam_1/AssignAssign:Generator/first/Generator/firstfully_connected/bias/Adam_1LGenerator/first/Generator/firstfully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:ђ
ш
?Generator/first/Generator/firstfully_connected/bias/Adam_1/readIdentity:Generator/first/Generator/firstfully_connected/bias/Adam_1*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
_output_shapes	
:ђ
ч
^Generator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB"ђ      *
dtype0*
_output_shapes
:
т
TGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB
 *    
Ћ
NGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zerosFill^Generator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorTGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros/Const*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*

index_type0* 
_output_shapes
:
ђђ
ђ
<Generator/second/Generator/secondfully_connected/kernel/Adam
VariableV2*
	container *
shape:
ђђ*
dtype0* 
_output_shapes
:
ђђ*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel
ч
CGenerator/second/Generator/secondfully_connected/kernel/Adam/AssignAssign<Generator/second/Generator/secondfully_connected/kernel/AdamNGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:
ђђ
ѓ
AGenerator/second/Generator/secondfully_connected/kernel/Adam/readIdentity<Generator/second/Generator/secondfully_connected/kernel/Adam*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:
ђђ
§
`Generator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB"ђ      *
dtype0*
_output_shapes
:
у
VGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Џ
PGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zerosFill`Generator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorVGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*

index_type0* 
_output_shapes
:
ђђ
ѓ
>Generator/second/Generator/secondfully_connected/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
ђђ*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
	container *
shape:
ђђ
Ђ
EGenerator/second/Generator/secondfully_connected/kernel/Adam_1/AssignAssign>Generator/second/Generator/secondfully_connected/kernel/Adam_1PGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(
є
CGenerator/second/Generator/secondfully_connected/kernel/Adam_1/readIdentity>Generator/second/Generator/secondfully_connected/kernel/Adam_1*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:
ђђ
т
LGenerator/second/Generator/secondfully_connected/bias/Adam/Initializer/zerosConst*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
Ы
:Generator/second/Generator/secondfully_connected/bias/Adam
VariableV2*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias
Ь
AGenerator/second/Generator/secondfully_connected/bias/Adam/AssignAssign:Generator/second/Generator/secondfully_connected/bias/AdamLGenerator/second/Generator/secondfully_connected/bias/Adam/Initializer/zeros*
_output_shapes	
:ђ*
use_locking(*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
validate_shape(
э
?Generator/second/Generator/secondfully_connected/bias/Adam/readIdentity:Generator/second/Generator/secondfully_connected/bias/Adam*
_output_shapes	
:ђ*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias
у
NGenerator/second/Generator/secondfully_connected/bias/Adam_1/Initializer/zerosConst*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
З
<Generator/second/Generator/secondfully_connected/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
	container *
shape:ђ
З
CGenerator/second/Generator/secondfully_connected/bias/Adam_1/AssignAssign<Generator/second/Generator/secondfully_connected/bias/Adam_1NGenerator/second/Generator/secondfully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:ђ
ч
AGenerator/second/Generator/secondfully_connected/bias/Adam_1/readIdentity<Generator/second/Generator/secondfully_connected/bias/Adam_1*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
_output_shapes	
:ђ
ж
NGenerator/second/Generator/secondbatch_normalized/gamma/Adam/Initializer/zerosConst*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
Ш
<Generator/second/Generator/secondbatch_normalized/gamma/Adam
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
	container *
shape:ђ
Ш
CGenerator/second/Generator/secondbatch_normalized/gamma/Adam/AssignAssign<Generator/second/Generator/secondbatch_normalized/gamma/AdamNGenerator/second/Generator/secondbatch_normalized/gamma/Adam/Initializer/zeros*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0
§
AGenerator/second/Generator/secondbatch_normalized/gamma/Adam/readIdentity<Generator/second/Generator/secondbatch_normalized/gamma/Adam*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
_output_shapes	
:ђ
в
PGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/Initializer/zerosConst*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
Э
>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
	container *
shape:ђ
Ч
EGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/AssignAssign>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1PGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:ђ
Ђ
CGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/readIdentity>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
_output_shapes	
:ђ
у
MGenerator/second/Generator/secondbatch_normalized/beta/Adam/Initializer/zerosConst*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
З
;Generator/second/Generator/secondbatch_normalized/beta/Adam
VariableV2*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name 
Ы
BGenerator/second/Generator/secondbatch_normalized/beta/Adam/AssignAssign;Generator/second/Generator/secondbatch_normalized/beta/AdamMGenerator/second/Generator/secondbatch_normalized/beta/Adam/Initializer/zeros*
_output_shapes	
:ђ*
use_locking(*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
validate_shape(
Щ
@Generator/second/Generator/secondbatch_normalized/beta/Adam/readIdentity;Generator/second/Generator/secondbatch_normalized/beta/Adam*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
_output_shapes	
:ђ
ж
OGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/Initializer/zerosConst*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
Ш
=Generator/second/Generator/secondbatch_normalized/beta/Adam_1
VariableV2*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name 
Э
DGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/AssignAssign=Generator/second/Generator/secondbatch_normalized/beta/Adam_1OGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/Initializer/zeros*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
■
BGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/readIdentity=Generator/second/Generator/secondbatch_normalized/beta/Adam_1*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
_output_shapes	
:ђ
э
\Generator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
р
RGenerator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ї
LGenerator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zerosFill\Generator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorRGenerator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros/Const*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*

index_type0* 
_output_shapes
:
ђђ
Ч
:Generator/third/Generator/thirdfully_connected/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
ђђ*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
	container *
shape:
ђђ
з
AGenerator/third/Generator/thirdfully_connected/kernel/Adam/AssignAssign:Generator/third/Generator/thirdfully_connected/kernel/AdamLGenerator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(*
T0
Ч
?Generator/third/Generator/thirdfully_connected/kernel/Adam/readIdentity:Generator/third/Generator/thirdfully_connected/kernel/Adam*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel* 
_output_shapes
:
ђђ
щ
^Generator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB"      
с
TGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Њ
NGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zerosFill^Generator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorTGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
ђђ*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*

index_type0
■
<Generator/third/Generator/thirdfully_connected/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
ђђ*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
	container *
shape:
ђђ
щ
CGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/AssignAssign<Generator/third/Generator/thirdfully_connected/kernel/Adam_1NGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel
ђ
AGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/readIdentity<Generator/third/Generator/thirdfully_connected/kernel/Adam_1*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel* 
_output_shapes
:
ђђ
р
JGenerator/third/Generator/thirdfully_connected/bias/Adam/Initializer/zerosConst*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
Ь
8Generator/third/Generator/thirdfully_connected/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
	container *
shape:ђ
Т
?Generator/third/Generator/thirdfully_connected/bias/Adam/AssignAssign8Generator/third/Generator/thirdfully_connected/bias/AdamJGenerator/third/Generator/thirdfully_connected/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias
ы
=Generator/third/Generator/thirdfully_connected/bias/Adam/readIdentity8Generator/third/Generator/thirdfully_connected/bias/Adam*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
_output_shapes	
:ђ*
T0
с
LGenerator/third/Generator/thirdfully_connected/bias/Adam_1/Initializer/zerosConst*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
­
:Generator/third/Generator/thirdfully_connected/bias/Adam_1
VariableV2*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias
В
AGenerator/third/Generator/thirdfully_connected/bias/Adam_1/AssignAssign:Generator/third/Generator/thirdfully_connected/bias/Adam_1LGenerator/third/Generator/thirdfully_connected/bias/Adam_1/Initializer/zeros*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0
ш
?Generator/third/Generator/thirdfully_connected/bias/Adam_1/readIdentity:Generator/third/Generator/thirdfully_connected/bias/Adam_1*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
_output_shapes	
:ђ
т
LGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/Initializer/zerosConst*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
Ы
:Generator/third/Generator/thirdbatch_normalized/gamma/Adam
VariableV2*
_output_shapes	
:ђ*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
	container *
shape:ђ*
dtype0
Ь
AGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/AssignAssign:Generator/third/Generator/thirdbatch_normalized/gamma/AdamLGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:ђ
э
?Generator/third/Generator/thirdbatch_normalized/gamma/Adam/readIdentity:Generator/third/Generator/thirdbatch_normalized/gamma/Adam*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
_output_shapes	
:ђ
у
NGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/Initializer/zerosConst*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
З
<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
	container *
shape:ђ
З
CGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/AssignAssign<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1NGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/Initializer/zeros*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0
ч
AGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/readIdentity<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1*
_output_shapes	
:ђ*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma
с
KGenerator/third/Generator/thirdbatch_normalized/beta/Adam/Initializer/zerosConst*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
­
9Generator/third/Generator/thirdbatch_normalized/beta/Adam
VariableV2*
_output_shapes	
:ђ*
shared_name *G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
	container *
shape:ђ*
dtype0
Ж
@Generator/third/Generator/thirdbatch_normalized/beta/Adam/AssignAssign9Generator/third/Generator/thirdbatch_normalized/beta/AdamKGenerator/third/Generator/thirdbatch_normalized/beta/Adam/Initializer/zeros*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
З
>Generator/third/Generator/thirdbatch_normalized/beta/Adam/readIdentity9Generator/third/Generator/thirdbatch_normalized/beta/Adam*
_output_shapes	
:ђ*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta
т
MGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/Initializer/zerosConst*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
Ы
;Generator/third/Generator/thirdbatch_normalized/beta/Adam_1
VariableV2*
_output_shapes	
:ђ*
shared_name *G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
	container *
shape:ђ*
dtype0
­
BGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/AssignAssign;Generator/third/Generator/thirdbatch_normalized/beta/Adam_1MGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
validate_shape(*
_output_shapes	
:ђ
Э
@Generator/third/Generator/thirdbatch_normalized/beta/Adam_1/readIdentity;Generator/third/Generator/thirdbatch_normalized/beta/Adam_1*
_output_shapes	
:ђ*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta
э
\Generator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
р
RGenerator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ї
LGenerator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zerosFill\Generator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorRGenerator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros/Const*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*

index_type0* 
_output_shapes
:
ђђ
Ч
:Generator/forth/Generator/forthfully_connected/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
ђђ*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
	container *
shape:
ђђ
з
AGenerator/forth/Generator/forthfully_connected/kernel/Adam/AssignAssign:Generator/forth/Generator/forthfully_connected/kernel/AdamLGenerator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(*
T0
Ч
?Generator/forth/Generator/forthfully_connected/kernel/Adam/readIdentity:Generator/forth/Generator/forthfully_connected/kernel/Adam* 
_output_shapes
:
ђђ*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel
щ
^Generator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
с
TGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Њ
NGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zerosFill^Generator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorTGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*

index_type0* 
_output_shapes
:
ђђ
■
<Generator/forth/Generator/forthfully_connected/kernel/Adam_1
VariableV2* 
_output_shapes
:
ђђ*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
	container *
shape:
ђђ*
dtype0
щ
CGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/AssignAssign<Generator/forth/Generator/forthfully_connected/kernel/Adam_1NGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
validate_shape(* 
_output_shapes
:
ђђ
ђ
AGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/readIdentity<Generator/forth/Generator/forthfully_connected/kernel/Adam_1*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:
ђђ
ь
ZGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB:ђ
П
PGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros/ConstConst*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
ђ
JGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zerosFillZGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros/shape_as_tensorPGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros/Const*
_output_shapes	
:ђ*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*

index_type0
Ь
8Generator/forth/Generator/forthfully_connected/bias/Adam
VariableV2*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name 
Т
?Generator/forth/Generator/forthfully_connected/bias/Adam/AssignAssign8Generator/forth/Generator/forthfully_connected/bias/AdamJGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
validate_shape(*
_output_shapes	
:ђ
ы
=Generator/forth/Generator/forthfully_connected/bias/Adam/readIdentity8Generator/forth/Generator/forthfully_connected/bias/Adam*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
_output_shapes	
:ђ
№
\Generator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB:ђ*
dtype0*
_output_shapes
:
▀
RGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB
 *    
є
LGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zerosFill\Generator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorRGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/Const*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*

index_type0*
_output_shapes	
:ђ
­
:Generator/forth/Generator/forthfully_connected/bias/Adam_1
VariableV2*
shared_name *F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ
В
AGenerator/forth/Generator/forthfully_connected/bias/Adam_1/AssignAssign:Generator/forth/Generator/forthfully_connected/bias/Adam_1LGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias
ш
?Generator/forth/Generator/forthfully_connected/bias/Adam_1/readIdentity:Generator/forth/Generator/forthfully_connected/bias/Adam_1*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
_output_shapes	
:ђ*
T0
ы
\Generator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB:ђ
р
RGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB
 *    *
dtype0*
_output_shapes
: 
ѕ
LGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zerosFill\Generator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/shape_as_tensorRGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/Const*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*

index_type0*
_output_shapes	
:ђ*
T0
Ы
:Generator/forth/Generator/forthbatch_normalized/gamma/Adam
VariableV2*
_output_shapes	
:ђ*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
	container *
shape:ђ*
dtype0
Ь
AGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/AssignAssign:Generator/forth/Generator/forthbatch_normalized/gamma/AdamLGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:ђ
э
?Generator/forth/Generator/forthbatch_normalized/gamma/Adam/readIdentity:Generator/forth/Generator/forthbatch_normalized/gamma/Adam*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
_output_shapes	
:ђ
з
^Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB:ђ*
dtype0*
_output_shapes
:
с
TGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB
 *    
ј
NGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zerosFill^Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros/shape_as_tensorTGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros/Const*
_output_shapes	
:ђ*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*

index_type0
З
<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1
VariableV2*
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
	container 
З
CGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/AssignAssign<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1NGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0
ч
AGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/readIdentity<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
_output_shapes	
:ђ*
T0
№
[Generator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros/shape_as_tensorConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB:ђ*
dtype0*
_output_shapes
:
▀
QGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros/ConstConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
ё
KGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zerosFill[Generator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros/shape_as_tensorQGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros/Const*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*

index_type0*
_output_shapes	
:ђ
­
9Generator/forth/Generator/forthbatch_normalized/beta/Adam
VariableV2*
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
	container 
Ж
@Generator/forth/Generator/forthbatch_normalized/beta/Adam/AssignAssign9Generator/forth/Generator/forthbatch_normalized/beta/AdamKGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0
З
>Generator/forth/Generator/forthbatch_normalized/beta/Adam/readIdentity9Generator/forth/Generator/forthbatch_normalized/beta/Adam*
_output_shapes	
:ђ*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta
ы
]Generator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros/shape_as_tensorConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB:ђ*
dtype0*
_output_shapes
:
р
SGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB
 *    *
dtype0
і
MGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zerosFill]Generator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros/shape_as_tensorSGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros/Const*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*

index_type0*
_output_shapes	
:ђ
Ы
;Generator/forth/Generator/forthbatch_normalized/beta/Adam_1
VariableV2*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name 
­
BGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/AssignAssign;Generator/forth/Generator/forthbatch_normalized/beta/Adam_1MGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros*
_output_shapes	
:ђ*
use_locking(*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
validate_shape(
Э
@Generator/forth/Generator/forthbatch_normalized/beta/Adam_1/readIdentity;Generator/forth/Generator/forthbatch_normalized/beta/Adam_1*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
_output_shapes	
:ђ*
T0
╣
=Generator/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*)
_class
loc:@Generator/dense/kernel*
valueB"     *
dtype0
Б
3Generator/dense/kernel/Adam/Initializer/zeros/ConstConst*)
_class
loc:@Generator/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Љ
-Generator/dense/kernel/Adam/Initializer/zerosFill=Generator/dense/kernel/Adam/Initializer/zeros/shape_as_tensor3Generator/dense/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
ђљ*
T0*)
_class
loc:@Generator/dense/kernel*

index_type0
Й
Generator/dense/kernel/Adam
VariableV2*)
_class
loc:@Generator/dense/kernel*
	container *
shape:
ђљ*
dtype0* 
_output_shapes
:
ђљ*
shared_name 
э
"Generator/dense/kernel/Adam/AssignAssignGenerator/dense/kernel/Adam-Generator/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@Generator/dense/kernel*
validate_shape(* 
_output_shapes
:
ђљ
Ъ
 Generator/dense/kernel/Adam/readIdentityGenerator/dense/kernel/Adam* 
_output_shapes
:
ђљ*
T0*)
_class
loc:@Generator/dense/kernel
╗
?Generator/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@Generator/dense/kernel*
valueB"     *
dtype0*
_output_shapes
:
Ц
5Generator/dense/kernel/Adam_1/Initializer/zeros/ConstConst*)
_class
loc:@Generator/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ќ
/Generator/dense/kernel/Adam_1/Initializer/zerosFill?Generator/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor5Generator/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*)
_class
loc:@Generator/dense/kernel*

index_type0* 
_output_shapes
:
ђљ
└
Generator/dense/kernel/Adam_1
VariableV2*)
_class
loc:@Generator/dense/kernel*
	container *
shape:
ђљ*
dtype0* 
_output_shapes
:
ђљ*
shared_name 
§
$Generator/dense/kernel/Adam_1/AssignAssignGenerator/dense/kernel/Adam_1/Generator/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@Generator/dense/kernel*
validate_shape(* 
_output_shapes
:
ђљ
Б
"Generator/dense/kernel/Adam_1/readIdentityGenerator/dense/kernel/Adam_1*
T0*)
_class
loc:@Generator/dense/kernel* 
_output_shapes
:
ђљ
Б
+Generator/dense/bias/Adam/Initializer/zerosConst*'
_class
loc:@Generator/dense/bias*
valueBљ*    *
dtype0*
_output_shapes	
:љ
░
Generator/dense/bias/Adam
VariableV2*
	container *
shape:љ*
dtype0*
_output_shapes	
:љ*
shared_name *'
_class
loc:@Generator/dense/bias
Ж
 Generator/dense/bias/Adam/AssignAssignGenerator/dense/bias/Adam+Generator/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes	
:љ
ћ
Generator/dense/bias/Adam/readIdentityGenerator/dense/bias/Adam*
T0*'
_class
loc:@Generator/dense/bias*
_output_shapes	
:љ
Ц
-Generator/dense/bias/Adam_1/Initializer/zerosConst*'
_class
loc:@Generator/dense/bias*
valueBљ*    *
dtype0*
_output_shapes	
:љ
▓
Generator/dense/bias/Adam_1
VariableV2*
shared_name *'
_class
loc:@Generator/dense/bias*
	container *
shape:љ*
dtype0*
_output_shapes	
:љ
­
"Generator/dense/bias/Adam_1/AssignAssignGenerator/dense/bias/Adam_1-Generator/dense/bias/Adam_1/Initializer/zeros*
_output_shapes	
:љ*
use_locking(*
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(
ў
 Generator/dense/bias/Adam_1/readIdentityGenerator/dense/bias/Adam_1*
T0*'
_class
loc:@Generator/dense/bias*
_output_shapes	
:љ
Y
Adam_1/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *иQ9
Q
Adam_1/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Q
Adam_1/beta2Const*
valueB
 *wЙ?*
dtype0*
_output_shapes
: 
S
Adam_1/epsilonConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
з
MAdam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdam	ApplyAdam5Generator/first/Generator/firstfully_connected/kernel:Generator/first/Generator/firstfully_connected/kernel/Adam<Generator/first/Generator/firstfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/control_dependency_1*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
use_nesterov( *
_output_shapes
:	dђ*
use_locking( 
Т
KAdam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdam	ApplyAdam3Generator/first/Generator/firstfully_connected/bias8Generator/first/Generator/firstfully_connected/bias/Adam:Generator/first/Generator/firstfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
use_nesterov( *
_output_shapes	
:ђ
ђ
OAdam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdam	ApplyAdam7Generator/second/Generator/secondfully_connected/kernel<Generator/second/Generator/secondfully_connected/kernel/Adam>Generator/second/Generator/secondfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsiloncgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
use_nesterov( * 
_output_shapes
:
ђђ
Ы
MAdam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdam	ApplyAdam5Generator/second/Generator/secondfully_connected/bias:Generator/second/Generator/secondfully_connected/bias/Adam<Generator/second/Generator/secondfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilondgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
use_nesterov( *
_output_shapes	
:ђ
Ѓ
OAdam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdam	ApplyAdam7Generator/second/Generator/secondbatch_normalized/gamma<Generator/second/Generator/secondbatch_normalized/gamma/Adam>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
use_nesterov( *
_output_shapes	
:ђ*
use_locking( 
Ч
NAdam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdam	ApplyAdam6Generator/second/Generator/secondbatch_normalized/beta;Generator/second/Generator/secondbatch_normalized/beta/Adam=Generator/second/Generator/secondbatch_normalized/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonigradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
use_nesterov( *
_output_shapes	
:ђ
З
MAdam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam	ApplyAdam5Generator/third/Generator/thirdfully_connected/kernel:Generator/third/Generator/thirdfully_connected/kernel/Adam<Generator/third/Generator/thirdfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
use_nesterov( * 
_output_shapes
:
ђђ
Т
KAdam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdam	ApplyAdam3Generator/third/Generator/thirdfully_connected/bias8Generator/third/Generator/thirdfully_connected/bias/Adam:Generator/third/Generator/thirdfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:ђ*
use_locking( *
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias
э
MAdam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdam	ApplyAdam5Generator/third/Generator/thirdbatch_normalized/gamma:Generator/third/Generator/thirdbatch_normalized/gamma/Adam<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
use_nesterov( *
_output_shapes	
:ђ*
use_locking( *
T0
­
LAdam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdam	ApplyAdam4Generator/third/Generator/thirdbatch_normalized/beta9Generator/third/Generator/thirdbatch_normalized/beta/Adam;Generator/third/Generator/thirdbatch_normalized/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonggradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
use_nesterov( *
_output_shapes	
:ђ
З
MAdam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdam	ApplyAdam5Generator/forth/Generator/forthfully_connected/kernel:Generator/forth/Generator/forthfully_connected/kernel/Adam<Generator/forth/Generator/forthfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency_1*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
use_nesterov( * 
_output_shapes
:
ђђ*
use_locking( *
T0
Т
KAdam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdam	ApplyAdam3Generator/forth/Generator/forthfully_connected/bias8Generator/forth/Generator/forthfully_connected/bias/Adam:Generator/forth/Generator/forthfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
use_nesterov( *
_output_shapes	
:ђ
э
MAdam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdam	ApplyAdam5Generator/forth/Generator/forthbatch_normalized/gamma:Generator/forth/Generator/forthbatch_normalized/gamma/Adam<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
use_nesterov( *
_output_shapes	
:ђ*
use_locking( 
­
LAdam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdam	ApplyAdam4Generator/forth/Generator/forthbatch_normalized/beta9Generator/forth/Generator/forthbatch_normalized/beta/Adam;Generator/forth/Generator/forthbatch_normalized/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonggradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependency*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
use_nesterov( *
_output_shapes	
:ђ*
use_locking( 
║
.Adam_1/update_Generator/dense/kernel/ApplyAdam	ApplyAdamGenerator/dense/kernelGenerator/dense/kernel/AdamGenerator/dense/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonBgradients_1/Generator/dense/MatMul_grad/tuple/control_dependency_1*
T0*)
_class
loc:@Generator/dense/kernel*
use_nesterov( * 
_output_shapes
:
ђљ*
use_locking( 
г
,Adam_1/update_Generator/dense/bias/ApplyAdam	ApplyAdamGenerator/dense/biasGenerator/dense/bias/AdamGenerator/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonCgradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependency_1*
T0*'
_class
loc:@Generator/dense/bias*
use_nesterov( *
_output_shapes	
:љ*
use_locking( 
║


Adam_1/mulMulbeta1_power_1/readAdam_1/beta1-^Adam_1/update_Generator/dense/bias/ApplyAdam/^Adam_1/update_Generator/dense/kernel/ApplyAdamL^Adam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdamO^Adam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdamN^Adam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam*'
_class
loc:@Generator/dense/bias*
_output_shapes
: *
T0
Ц
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
use_locking( *
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes
: 
╝

Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2-^Adam_1/update_Generator/dense/bias/ApplyAdam/^Adam_1/update_Generator/dense/kernel/ApplyAdamL^Adam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdamO^Adam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdamN^Adam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam*
T0*'
_class
loc:@Generator/dense/bias*
_output_shapes
: 
Е
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
use_locking( *
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes
: 
ь	
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_1-^Adam_1/update_Generator/dense/bias/ApplyAdam/^Adam_1/update_Generator/dense/kernel/ApplyAdamL^Adam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdamO^Adam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdamN^Adam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam
g
Merge/MergeSummaryMergeSummarydiscriminator_lossgenerator_loss*
N*
_output_shapes
: "џУ-░b     ╣рx6	Еs ▒§оAJБ┼
Эо
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	ђљ
Ь
	ApplyAdam
var"Tђ	
m"Tђ	
v"Tђ
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"Tђ" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"Tђ

value"T

output_ref"Tђ"	
Ttype"
validate_shapebool("
use_lockingbool(ў
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
,
Log
x"T
y"T"
Ttype:

2
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	љ
Ї
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
=
Mul
x"T
y"T
z"T"
Ttype:
2	љ
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
Ї
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	ѕ
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
.
Rsqrt
x"T
y"T"
Ttype:

2
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
0
Sigmoid
x"T
y"T"
Ttype:

2
=
SigmoidGrad
y"T
dy"T
z"T"
Ttype:

2
:
Sub
x"T
y"T
z"T"
Ttype:
2	
ї
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtypeђ"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ѕ*1.12.02v1.12.0-0-ga6d8ffae09╣ж
r
Generator/noisePlaceholder*
dtype0*'
_output_shapes
:         d*
shape:         d
ы
VGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/shapeConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB"d   ђ   *
dtype0*
_output_shapes
:
с
TGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/minConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB
 *_&Й*
dtype0*
_output_shapes
: 
с
TGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB
 *_&>
р
^Generator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformVGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/shape*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
seed2*
dtype0*
_output_shapes
:	dђ*

seed
Ы
TGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/subSubTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/maxTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel
Ё
TGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/mulMul^Generator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniformTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/sub*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
_output_shapes
:	dђ
э
PGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniformAddTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/mulTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
_output_shapes
:	dђ
ш
5Generator/first/Generator/firstfully_connected/kernel
VariableV2*
shared_name *H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
	container *
shape:	dђ*
dtype0*
_output_shapes
:	dђ
В
<Generator/first/Generator/firstfully_connected/kernel/AssignAssign5Generator/first/Generator/firstfully_connected/kernelPGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
validate_shape(*
_output_shapes
:	dђ
ы
:Generator/first/Generator/firstfully_connected/kernel/readIdentity5Generator/first/Generator/firstfully_connected/kernel*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
_output_shapes
:	dђ
▄
EGenerator/first/Generator/firstfully_connected/bias/Initializer/zerosConst*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
ж
3Generator/first/Generator/firstfully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
	container *
shape:ђ
О
:Generator/first/Generator/firstfully_connected/bias/AssignAssign3Generator/first/Generator/firstfully_connected/biasEGenerator/first/Generator/firstfully_connected/bias/Initializer/zeros*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
у
8Generator/first/Generator/firstfully_connected/bias/readIdentity3Generator/first/Generator/firstfully_connected/bias*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
_output_shapes	
:ђ
т
5Generator/first/Generator/firstfully_connected/MatMulMatMulGenerator/noise:Generator/first/Generator/firstfully_connected/kernel/read*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b( *
T0
Ч
6Generator/first/Generator/firstfully_connected/BiasAddBiasAdd5Generator/first/Generator/firstfully_connected/MatMul8Generator/first/Generator/firstfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         ђ
t
/Generator/first/Generator/firstleaky_relu/alphaConst*
_output_shapes
: *
valueB
 *═╠L>*
dtype0
л
-Generator/first/Generator/firstleaky_relu/mulMul/Generator/first/Generator/firstleaky_relu/alpha6Generator/first/Generator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
╬
)Generator/first/Generator/firstleaky_reluMaximum-Generator/first/Generator/firstleaky_relu/mul6Generator/first/Generator/firstfully_connected/BiasAdd*(
_output_shapes
:         ђ*
T0
ш
XGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/shapeConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB"ђ      *
dtype0*
_output_shapes
:
у
VGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/minConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB
 *   Й*
dtype0*
_output_shapes
: 
у
VGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/maxConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB
 *   >*
dtype0*
_output_shapes
: 
У
`Generator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformXGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
ђђ*

seed*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
seed2
Щ
VGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/subSubVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/maxVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/min*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
_output_shapes
: 
ј
VGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/mulMul`Generator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/RandomUniformVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/sub*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:
ђђ
ђ
RGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniformAddVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/mulVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/min*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:
ђђ
ч
7Generator/second/Generator/secondfully_connected/kernel
VariableV2*
dtype0* 
_output_shapes
:
ђђ*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
	container *
shape:
ђђ
ш
>Generator/second/Generator/secondfully_connected/kernel/AssignAssign7Generator/second/Generator/secondfully_connected/kernelRGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(*
T0
Э
<Generator/second/Generator/secondfully_connected/kernel/readIdentity7Generator/second/Generator/secondfully_connected/kernel*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:
ђђ
Я
GGenerator/second/Generator/secondfully_connected/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:ђ*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
valueBђ*    
ь
5Generator/second/Generator/secondfully_connected/bias
VariableV2*
_output_shapes	
:ђ*
shared_name *H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
	container *
shape:ђ*
dtype0
▀
<Generator/second/Generator/secondfully_connected/bias/AssignAssign5Generator/second/Generator/secondfully_connected/biasGGenerator/second/Generator/secondfully_connected/bias/Initializer/zeros*
_output_shapes	
:ђ*
use_locking(*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
validate_shape(
ь
:Generator/second/Generator/secondfully_connected/bias/readIdentity5Generator/second/Generator/secondfully_connected/bias*
_output_shapes	
:ђ*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias
Ѓ
7Generator/second/Generator/secondfully_connected/MatMulMatMul)Generator/first/Generator/firstleaky_relu<Generator/second/Generator/secondfully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:         ђ*
transpose_a( 
ѓ
8Generator/second/Generator/secondfully_connected/BiasAddBiasAdd7Generator/second/Generator/secondfully_connected/MatMul:Generator/second/Generator/secondfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         ђ
с
HGenerator/second/Generator/secondbatch_normalized/gamma/Initializer/onesConst*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
valueBђ*  ђ?*
dtype0*
_output_shapes	
:ђ
ы
7Generator/second/Generator/secondbatch_normalized/gamma
VariableV2*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name 
Т
>Generator/second/Generator/secondbatch_normalized/gamma/AssignAssign7Generator/second/Generator/secondbatch_normalized/gammaHGenerator/second/Generator/secondbatch_normalized/gamma/Initializer/ones*
use_locking(*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:ђ
з
<Generator/second/Generator/secondbatch_normalized/gamma/readIdentity7Generator/second/Generator/secondbatch_normalized/gamma*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
_output_shapes	
:ђ
Р
HGenerator/second/Generator/secondbatch_normalized/beta/Initializer/zerosConst*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
№
6Generator/second/Generator/secondbatch_normalized/beta
VariableV2*
shared_name *I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ
с
=Generator/second/Generator/secondbatch_normalized/beta/AssignAssign6Generator/second/Generator/secondbatch_normalized/betaHGenerator/second/Generator/secondbatch_normalized/beta/Initializer/zeros*
use_locking(*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
validate_shape(*
_output_shapes	
:ђ
­
;Generator/second/Generator/secondbatch_normalized/beta/readIdentity6Generator/second/Generator/secondbatch_normalized/beta*
_output_shapes	
:ђ*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta
­
OGenerator/second/Generator/secondbatch_normalized/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:ђ*P
_classF
DBloc:@Generator/second/Generator/secondbatch_normalized/moving_mean*
valueBђ*    
§
=Generator/second/Generator/secondbatch_normalized/moving_mean
VariableV2*P
_classF
DBloc:@Generator/second/Generator/secondbatch_normalized/moving_mean*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name 
 
DGenerator/second/Generator/secondbatch_normalized/moving_mean/AssignAssign=Generator/second/Generator/secondbatch_normalized/moving_meanOGenerator/second/Generator/secondbatch_normalized/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0*P
_classF
DBloc:@Generator/second/Generator/secondbatch_normalized/moving_mean
Ё
BGenerator/second/Generator/secondbatch_normalized/moving_mean/readIdentity=Generator/second/Generator/secondbatch_normalized/moving_mean*
_output_shapes	
:ђ*
T0*P
_classF
DBloc:@Generator/second/Generator/secondbatch_normalized/moving_mean
э
RGenerator/second/Generator/secondbatch_normalized/moving_variance/Initializer/onesConst*
_output_shapes	
:ђ*T
_classJ
HFloc:@Generator/second/Generator/secondbatch_normalized/moving_variance*
valueBђ*  ђ?*
dtype0
Ё
AGenerator/second/Generator/secondbatch_normalized/moving_variance
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *T
_classJ
HFloc:@Generator/second/Generator/secondbatch_normalized/moving_variance*
	container *
shape:ђ
ј
HGenerator/second/Generator/secondbatch_normalized/moving_variance/AssignAssignAGenerator/second/Generator/secondbatch_normalized/moving_varianceRGenerator/second/Generator/secondbatch_normalized/moving_variance/Initializer/ones*
use_locking(*
T0*T
_classJ
HFloc:@Generator/second/Generator/secondbatch_normalized/moving_variance*
validate_shape(*
_output_shapes	
:ђ
Љ
FGenerator/second/Generator/secondbatch_normalized/moving_variance/readIdentityAGenerator/second/Generator/secondbatch_normalized/moving_variance*
T0*T
_classJ
HFloc:@Generator/second/Generator/secondbatch_normalized/moving_variance*
_output_shapes	
:ђ
є
AGenerator/second/Generator/secondbatch_normalized/batchnorm/add/yConst*
valueB
 *oЃ:*
dtype0*
_output_shapes
: 
э
?Generator/second/Generator/secondbatch_normalized/batchnorm/addAddFGenerator/second/Generator/secondbatch_normalized/moving_variance/readAGenerator/second/Generator/secondbatch_normalized/batchnorm/add/y*
T0*
_output_shapes	
:ђ
▒
AGenerator/second/Generator/secondbatch_normalized/batchnorm/RsqrtRsqrt?Generator/second/Generator/secondbatch_normalized/batchnorm/add*
T0*
_output_shapes	
:ђ
ь
?Generator/second/Generator/secondbatch_normalized/batchnorm/mulMulAGenerator/second/Generator/secondbatch_normalized/batchnorm/Rsqrt<Generator/second/Generator/secondbatch_normalized/gamma/read*
_output_shapes	
:ђ*
T0
Ш
AGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_1Mul8Generator/second/Generator/secondfully_connected/BiasAdd?Generator/second/Generator/secondbatch_normalized/batchnorm/mul*(
_output_shapes
:         ђ*
T0
з
AGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_2MulBGenerator/second/Generator/secondbatch_normalized/moving_mean/read?Generator/second/Generator/secondbatch_normalized/batchnorm/mul*
_output_shapes	
:ђ*
T0
В
?Generator/second/Generator/secondbatch_normalized/batchnorm/subSub;Generator/second/Generator/secondbatch_normalized/beta/readAGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_2*
T0*
_output_shapes	
:ђ
 
AGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1AddAGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_1?Generator/second/Generator/secondbatch_normalized/batchnorm/sub*
T0*(
_output_shapes
:         ђ
v
1Generator/second/Generator/secondleaky_relu/alphaConst*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
▀
/Generator/second/Generator/secondleaky_relu/mulMul1Generator/second/Generator/secondleaky_relu/alphaAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:         ђ
П
+Generator/second/Generator/secondleaky_reluMaximum/Generator/second/Generator/secondleaky_relu/mulAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*(
_output_shapes
:         ђ*
T0
ы
VGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB"      *
dtype0
с
TGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB
 *зхй*
dtype0
с
TGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/maxConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB
 *зх=*
dtype0*
_output_shapes
: 
Р
^Generator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformVGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/shape*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
seed2B*
dtype0* 
_output_shapes
:
ђђ*

seed
Ы
TGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/subSubTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/maxTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
_output_shapes
: 
є
TGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/mulMul^Generator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/RandomUniformTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
ђђ*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel
Э
PGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniformAddTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/mulTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel* 
_output_shapes
:
ђђ
э
5Generator/third/Generator/thirdfully_connected/kernel
VariableV2*
dtype0* 
_output_shapes
:
ђђ*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
	container *
shape:
ђђ
ь
<Generator/third/Generator/thirdfully_connected/kernel/AssignAssign5Generator/third/Generator/thirdfully_connected/kernelPGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(
Ы
:Generator/third/Generator/thirdfully_connected/kernel/readIdentity5Generator/third/Generator/thirdfully_connected/kernel* 
_output_shapes
:
ђђ*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel
▄
EGenerator/third/Generator/thirdfully_connected/bias/Initializer/zerosConst*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
ж
3Generator/third/Generator/thirdfully_connected/bias
VariableV2*
shared_name *F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ
О
:Generator/third/Generator/thirdfully_connected/bias/AssignAssign3Generator/third/Generator/thirdfully_connected/biasEGenerator/third/Generator/thirdfully_connected/bias/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
validate_shape(*
_output_shapes	
:ђ
у
8Generator/third/Generator/thirdfully_connected/bias/readIdentity3Generator/third/Generator/thirdfully_connected/bias*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
_output_shapes	
:ђ
Ђ
5Generator/third/Generator/thirdfully_connected/MatMulMatMul+Generator/second/Generator/secondleaky_relu:Generator/third/Generator/thirdfully_connected/kernel/read*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b( *
T0
Ч
6Generator/third/Generator/thirdfully_connected/BiasAddBiasAdd5Generator/third/Generator/thirdfully_connected/MatMul8Generator/third/Generator/thirdfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         ђ
▀
FGenerator/third/Generator/thirdbatch_normalized/gamma/Initializer/onesConst*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
valueBђ*  ђ?*
dtype0*
_output_shapes	
:ђ
ь
5Generator/third/Generator/thirdbatch_normalized/gamma
VariableV2*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ
я
<Generator/third/Generator/thirdbatch_normalized/gamma/AssignAssign5Generator/third/Generator/thirdbatch_normalized/gammaFGenerator/third/Generator/thirdbatch_normalized/gamma/Initializer/ones*
use_locking(*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:ђ
ь
:Generator/third/Generator/thirdbatch_normalized/gamma/readIdentity5Generator/third/Generator/thirdbatch_normalized/gamma*
_output_shapes	
:ђ*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma
я
FGenerator/third/Generator/thirdbatch_normalized/beta/Initializer/zerosConst*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
в
4Generator/third/Generator/thirdbatch_normalized/beta
VariableV2*
shared_name *G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ
█
;Generator/third/Generator/thirdbatch_normalized/beta/AssignAssign4Generator/third/Generator/thirdbatch_normalized/betaFGenerator/third/Generator/thirdbatch_normalized/beta/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
validate_shape(*
_output_shapes	
:ђ
Ж
9Generator/third/Generator/thirdbatch_normalized/beta/readIdentity4Generator/third/Generator/thirdbatch_normalized/beta*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
_output_shapes	
:ђ
В
MGenerator/third/Generator/thirdbatch_normalized/moving_mean/Initializer/zerosConst*
_output_shapes	
:ђ*N
_classD
B@loc:@Generator/third/Generator/thirdbatch_normalized/moving_mean*
valueBђ*    *
dtype0
щ
;Generator/third/Generator/thirdbatch_normalized/moving_mean
VariableV2*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *N
_classD
B@loc:@Generator/third/Generator/thirdbatch_normalized/moving_mean
э
BGenerator/third/Generator/thirdbatch_normalized/moving_mean/AssignAssign;Generator/third/Generator/thirdbatch_normalized/moving_meanMGenerator/third/Generator/thirdbatch_normalized/moving_mean/Initializer/zeros*
use_locking(*
T0*N
_classD
B@loc:@Generator/third/Generator/thirdbatch_normalized/moving_mean*
validate_shape(*
_output_shapes	
:ђ
 
@Generator/third/Generator/thirdbatch_normalized/moving_mean/readIdentity;Generator/third/Generator/thirdbatch_normalized/moving_mean*
_output_shapes	
:ђ*
T0*N
_classD
B@loc:@Generator/third/Generator/thirdbatch_normalized/moving_mean
з
PGenerator/third/Generator/thirdbatch_normalized/moving_variance/Initializer/onesConst*R
_classH
FDloc:@Generator/third/Generator/thirdbatch_normalized/moving_variance*
valueBђ*  ђ?*
dtype0*
_output_shapes	
:ђ
Ђ
?Generator/third/Generator/thirdbatch_normalized/moving_variance
VariableV2*
shared_name *R
_classH
FDloc:@Generator/third/Generator/thirdbatch_normalized/moving_variance*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ
є
FGenerator/third/Generator/thirdbatch_normalized/moving_variance/AssignAssign?Generator/third/Generator/thirdbatch_normalized/moving_variancePGenerator/third/Generator/thirdbatch_normalized/moving_variance/Initializer/ones*
use_locking(*
T0*R
_classH
FDloc:@Generator/third/Generator/thirdbatch_normalized/moving_variance*
validate_shape(*
_output_shapes	
:ђ
І
DGenerator/third/Generator/thirdbatch_normalized/moving_variance/readIdentity?Generator/third/Generator/thirdbatch_normalized/moving_variance*
T0*R
_classH
FDloc:@Generator/third/Generator/thirdbatch_normalized/moving_variance*
_output_shapes	
:ђ
ё
?Generator/third/Generator/thirdbatch_normalized/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *oЃ:
ы
=Generator/third/Generator/thirdbatch_normalized/batchnorm/addAddDGenerator/third/Generator/thirdbatch_normalized/moving_variance/read?Generator/third/Generator/thirdbatch_normalized/batchnorm/add/y*
T0*
_output_shapes	
:ђ
Г
?Generator/third/Generator/thirdbatch_normalized/batchnorm/RsqrtRsqrt=Generator/third/Generator/thirdbatch_normalized/batchnorm/add*
T0*
_output_shapes	
:ђ
у
=Generator/third/Generator/thirdbatch_normalized/batchnorm/mulMul?Generator/third/Generator/thirdbatch_normalized/batchnorm/Rsqrt:Generator/third/Generator/thirdbatch_normalized/gamma/read*
T0*
_output_shapes	
:ђ
­
?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1Mul6Generator/third/Generator/thirdfully_connected/BiasAdd=Generator/third/Generator/thirdbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:         ђ
ь
?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2Mul@Generator/third/Generator/thirdbatch_normalized/moving_mean/read=Generator/third/Generator/thirdbatch_normalized/batchnorm/mul*
T0*
_output_shapes	
:ђ
Т
=Generator/third/Generator/thirdbatch_normalized/batchnorm/subSub9Generator/third/Generator/thirdbatch_normalized/beta/read?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2*
T0*
_output_shapes	
:ђ
щ
?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1Add?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1=Generator/third/Generator/thirdbatch_normalized/batchnorm/sub*
T0*(
_output_shapes
:         ђ
t
/Generator/third/Generator/thirdleaky_relu/alphaConst*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
┘
-Generator/third/Generator/thirdleaky_relu/mulMul/Generator/third/Generator/thirdleaky_relu/alpha?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:         ђ
О
)Generator/third/Generator/thirdleaky_reluMaximum-Generator/third/Generator/thirdleaky_relu/mul?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*(
_output_shapes
:         ђ*
T0
ы
VGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/shapeConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
с
TGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/minConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB
 *  ђй*
dtype0*
_output_shapes
: 
с
TGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/maxConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB
 *  ђ=*
dtype0*
_output_shapes
: 
Р
^Generator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformVGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/shape*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
seed2m*
dtype0* 
_output_shapes
:
ђђ*

seed
Ы
TGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/subSubTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/maxTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
_output_shapes
: 
є
TGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/mulMul^Generator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/RandomUniformTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/sub*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:
ђђ
Э
PGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniformAddTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/mulTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:
ђђ
э
5Generator/forth/Generator/forthfully_connected/kernel
VariableV2*
shape:
ђђ*
dtype0* 
_output_shapes
:
ђђ*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
	container 
ь
<Generator/forth/Generator/forthfully_connected/kernel/AssignAssign5Generator/forth/Generator/forthfully_connected/kernelPGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
validate_shape(* 
_output_shapes
:
ђђ
Ы
:Generator/forth/Generator/forthfully_connected/kernel/readIdentity5Generator/forth/Generator/forthfully_connected/kernel*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:
ђђ
У
UGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros/shape_as_tensorConst*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB:ђ*
dtype0*
_output_shapes
:
п
KGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros/ConstConst*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
ы
EGenerator/forth/Generator/forthfully_connected/bias/Initializer/zerosFillUGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros/shape_as_tensorKGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros/Const*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*

index_type0*
_output_shapes	
:ђ
ж
3Generator/forth/Generator/forthfully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
	container *
shape:ђ
О
:Generator/forth/Generator/forthfully_connected/bias/AssignAssign3Generator/forth/Generator/forthfully_connected/biasEGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
validate_shape(*
_output_shapes	
:ђ
у
8Generator/forth/Generator/forthfully_connected/bias/readIdentity3Generator/forth/Generator/forthfully_connected/bias*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
_output_shapes	
:ђ*
T0
 
5Generator/forth/Generator/forthfully_connected/MatMulMatMul)Generator/third/Generator/thirdleaky_relu:Generator/forth/Generator/forthfully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:         ђ*
transpose_a( 
Ч
6Generator/forth/Generator/forthfully_connected/BiasAddBiasAdd5Generator/forth/Generator/forthfully_connected/MatMul8Generator/forth/Generator/forthfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         ђ
в
VGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones/shape_as_tensorConst*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB:ђ*
dtype0*
_output_shapes
:
█
LGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones/ConstConst*
dtype0*
_output_shapes
: *H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB
 *  ђ?
Ш
FGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/onesFillVGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones/shape_as_tensorLGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones/Const*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*

index_type0*
_output_shapes	
:ђ*
T0
ь
5Generator/forth/Generator/forthbatch_normalized/gamma
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
	container *
shape:ђ
я
<Generator/forth/Generator/forthbatch_normalized/gamma/AssignAssign5Generator/forth/Generator/forthbatch_normalized/gammaFGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:ђ
ь
:Generator/forth/Generator/forthbatch_normalized/gamma/readIdentity5Generator/forth/Generator/forthbatch_normalized/gamma*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
_output_shapes	
:ђ
Ж
VGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros/shape_as_tensorConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB:ђ*
dtype0*
_output_shapes
:
┌
LGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros/ConstConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
ш
FGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zerosFillVGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros/shape_as_tensorLGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros/Const*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*

index_type0*
_output_shapes	
:ђ
в
4Generator/forth/Generator/forthbatch_normalized/beta
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
	container *
shape:ђ
█
;Generator/forth/Generator/forthbatch_normalized/beta/AssignAssign4Generator/forth/Generator/forthbatch_normalized/betaFGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta
Ж
9Generator/forth/Generator/forthbatch_normalized/beta/readIdentity4Generator/forth/Generator/forthbatch_normalized/beta*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
_output_shapes	
:ђ
Э
]Generator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros/shape_as_tensorConst*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
valueB:ђ*
dtype0*
_output_shapes
:
У
SGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros/ConstConst*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
Љ
MGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zerosFill]Generator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros/shape_as_tensorSGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros/Const*
T0*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*

index_type0*
_output_shapes	
:ђ
щ
;Generator/forth/Generator/forthbatch_normalized/moving_mean
VariableV2*
shared_name *N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ
э
BGenerator/forth/Generator/forthbatch_normalized/moving_mean/AssignAssign;Generator/forth/Generator/forthbatch_normalized/moving_meanMGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros*
T0*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
 
@Generator/forth/Generator/forthbatch_normalized/moving_mean/readIdentity;Generator/forth/Generator/forthbatch_normalized/moving_mean*
T0*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
_output_shapes	
:ђ
 
`Generator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones/shape_as_tensorConst*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
valueB:ђ*
dtype0*
_output_shapes
:
№
VGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones/ConstConst*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
ъ
PGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/onesFill`Generator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones/shape_as_tensorVGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones/Const*
T0*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*

index_type0*
_output_shapes	
:ђ
Ђ
?Generator/forth/Generator/forthbatch_normalized/moving_variance
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
	container *
shape:ђ
є
FGenerator/forth/Generator/forthbatch_normalized/moving_variance/AssignAssign?Generator/forth/Generator/forthbatch_normalized/moving_variancePGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance
І
DGenerator/forth/Generator/forthbatch_normalized/moving_variance/readIdentity?Generator/forth/Generator/forthbatch_normalized/moving_variance*
T0*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
_output_shapes	
:ђ
ё
?Generator/forth/Generator/forthbatch_normalized/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *oЃ:*
dtype0
ы
=Generator/forth/Generator/forthbatch_normalized/batchnorm/addAddDGenerator/forth/Generator/forthbatch_normalized/moving_variance/read?Generator/forth/Generator/forthbatch_normalized/batchnorm/add/y*
T0*
_output_shapes	
:ђ
Г
?Generator/forth/Generator/forthbatch_normalized/batchnorm/RsqrtRsqrt=Generator/forth/Generator/forthbatch_normalized/batchnorm/add*
T0*
_output_shapes	
:ђ
у
=Generator/forth/Generator/forthbatch_normalized/batchnorm/mulMul?Generator/forth/Generator/forthbatch_normalized/batchnorm/Rsqrt:Generator/forth/Generator/forthbatch_normalized/gamma/read*
_output_shapes	
:ђ*
T0
­
?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1Mul6Generator/forth/Generator/forthfully_connected/BiasAdd=Generator/forth/Generator/forthbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:         ђ
ь
?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2Mul@Generator/forth/Generator/forthbatch_normalized/moving_mean/read=Generator/forth/Generator/forthbatch_normalized/batchnorm/mul*
T0*
_output_shapes	
:ђ
Т
=Generator/forth/Generator/forthbatch_normalized/batchnorm/subSub9Generator/forth/Generator/forthbatch_normalized/beta/read?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2*
T0*
_output_shapes	
:ђ
щ
?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1Add?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1=Generator/forth/Generator/forthbatch_normalized/batchnorm/sub*(
_output_shapes
:         ђ*
T0
t
/Generator/forth/Generator/forthleaky_relu/alphaConst*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
┘
-Generator/forth/Generator/forthleaky_relu/mulMul/Generator/forth/Generator/forthleaky_relu/alpha?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*(
_output_shapes
:         ђ*
T0
О
)Generator/forth/Generator/forthleaky_reluMaximum-Generator/forth/Generator/forthleaky_relu/mul?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*(
_output_shapes
:         ђ*
T0
│
7Generator/dense/kernel/Initializer/random_uniform/shapeConst*)
_class
loc:@Generator/dense/kernel*
valueB"     *
dtype0*
_output_shapes
:
Ц
5Generator/dense/kernel/Initializer/random_uniform/minConst*)
_class
loc:@Generator/dense/kernel*
valueB
 *zшkй*
dtype0*
_output_shapes
: 
Ц
5Generator/dense/kernel/Initializer/random_uniform/maxConst*)
_class
loc:@Generator/dense/kernel*
valueB
 *zшk=*
dtype0*
_output_shapes
: 
є
?Generator/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform7Generator/dense/kernel/Initializer/random_uniform/shape*

seed*
T0*)
_class
loc:@Generator/dense/kernel*
seed2б*
dtype0* 
_output_shapes
:
ђљ
Ш
5Generator/dense/kernel/Initializer/random_uniform/subSub5Generator/dense/kernel/Initializer/random_uniform/max5Generator/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@Generator/dense/kernel*
_output_shapes
: 
і
5Generator/dense/kernel/Initializer/random_uniform/mulMul?Generator/dense/kernel/Initializer/random_uniform/RandomUniform5Generator/dense/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@Generator/dense/kernel* 
_output_shapes
:
ђљ
Ч
1Generator/dense/kernel/Initializer/random_uniformAdd5Generator/dense/kernel/Initializer/random_uniform/mul5Generator/dense/kernel/Initializer/random_uniform/min* 
_output_shapes
:
ђљ*
T0*)
_class
loc:@Generator/dense/kernel
╣
Generator/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
ђљ*
shared_name *)
_class
loc:@Generator/dense/kernel*
	container *
shape:
ђљ
ы
Generator/dense/kernel/AssignAssignGenerator/dense/kernel1Generator/dense/kernel/Initializer/random_uniform*)
_class
loc:@Generator/dense/kernel*
validate_shape(* 
_output_shapes
:
ђљ*
use_locking(*
T0
Ћ
Generator/dense/kernel/readIdentityGenerator/dense/kernel*)
_class
loc:@Generator/dense/kernel* 
_output_shapes
:
ђљ*
T0
ъ
&Generator/dense/bias/Initializer/zerosConst*'
_class
loc:@Generator/dense/bias*
valueBљ*    *
dtype0*
_output_shapes	
:љ
Ф
Generator/dense/bias
VariableV2*
dtype0*
_output_shapes	
:љ*
shared_name *'
_class
loc:@Generator/dense/bias*
	container *
shape:љ
█
Generator/dense/bias/AssignAssignGenerator/dense/bias&Generator/dense/bias/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes	
:љ
і
Generator/dense/bias/readIdentityGenerator/dense/bias*
_output_shapes	
:љ*
T0*'
_class
loc:@Generator/dense/bias
┴
Generator/dense/MatMulMatMul)Generator/forth/Generator/forthleaky_reluGenerator/dense/kernel/read*(
_output_shapes
:         љ*
transpose_a( *
transpose_b( *
T0
Ъ
Generator/dense/BiasAddBiasAddGenerator/dense/MatMulGenerator/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         љ
b
Generator/TanhTanhGenerator/dense/BiasAdd*
T0*(
_output_shapes
:         љ
w
Discriminator/realPlaceholder*
shape:         љ*
dtype0*(
_output_shapes
:         љ
Ђ
^Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/shapeConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
з
\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB
 *HYІй*
dtype0
з
\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/maxConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB
 *HYІ=*
dtype0*
_output_shapes
: 
ч
fDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniform^Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
љђ*

seed*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
seed2┤
њ
\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/subSub\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/max\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel
д
\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/mulMulfDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniform\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
љђ*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel
ў
XDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniformAdd\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/mul\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/min* 
_output_shapes
:
љђ*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel
Є
=Discriminator/first/Discriminator/firstfully_connected/kernel
VariableV2*
dtype0* 
_output_shapes
:
љђ*
shared_name *P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
	container *
shape:
љђ
Ї
DDiscriminator/first/Discriminator/firstfully_connected/kernel/AssignAssign=Discriminator/first/Discriminator/firstfully_connected/kernelXDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
љђ*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel
і
BDiscriminator/first/Discriminator/firstfully_connected/kernel/readIdentity=Discriminator/first/Discriminator/firstfully_connected/kernel*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel* 
_output_shapes
:
љђ
В
MDiscriminator/first/Discriminator/firstfully_connected/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:ђ*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueBђ*    
щ
;Discriminator/first/Discriminator/firstfully_connected/bias
VariableV2*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias
э
BDiscriminator/first/Discriminator/firstfully_connected/bias/AssignAssign;Discriminator/first/Discriminator/firstfully_connected/biasMDiscriminator/first/Discriminator/firstfully_connected/bias/Initializer/zeros*
_output_shapes	
:ђ*
use_locking(*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(
 
@Discriminator/first/Discriminator/firstfully_connected/bias/readIdentity;Discriminator/first/Discriminator/firstfully_connected/bias*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes	
:ђ
Э
=Discriminator/first/Discriminator/firstfully_connected/MatMulMatMulDiscriminator/realBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b( 
ћ
>Discriminator/first/Discriminator/firstfully_connected/BiasAddBiasAdd=Discriminator/first/Discriminator/firstfully_connected/MatMul@Discriminator/first/Discriminator/firstfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         ђ
|
7Discriminator/first/Discriminator/firstleaky_relu/alphaConst*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
У
5Discriminator/first/Discriminator/firstleaky_relu/mulMul7Discriminator/first/Discriminator/firstleaky_relu/alpha>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
Т
1Discriminator/first/Discriminator/firstleaky_reluMaximum5Discriminator/first/Discriminator/firstleaky_relu/mul>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*(
_output_shapes
:         ђ*
T0
Ё
`Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB"      
э
^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/minConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB
 *зхй*
dtype0*
_output_shapes
: 
э
^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/maxConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB
 *зх=*
dtype0*
_output_shapes
: 
Ђ
hDiscriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniform`Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/shape*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
seed2К*
dtype0* 
_output_shapes
:
ђђ*

seed
џ
^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/subSub^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/max^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/min*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
_output_shapes
: 
«
^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/mulMulhDiscriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/RandomUniform^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/sub*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel* 
_output_shapes
:
ђђ
а
ZDiscriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniformAdd^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/mul^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/min*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel* 
_output_shapes
:
ђђ
І
?Discriminator/second/Discriminator/secondfully_connected/kernel
VariableV2*
shared_name *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
	container *
shape:
ђђ*
dtype0* 
_output_shapes
:
ђђ
Ћ
FDiscriminator/second/Discriminator/secondfully_connected/kernel/AssignAssign?Discriminator/second/Discriminator/secondfully_connected/kernelZDiscriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(
љ
DDiscriminator/second/Discriminator/secondfully_connected/kernel/readIdentity?Discriminator/second/Discriminator/secondfully_connected/kernel* 
_output_shapes
:
ђђ*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel
­
ODiscriminator/second/Discriminator/secondfully_connected/bias/Initializer/zerosConst*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
§
=Discriminator/second/Discriminator/secondfully_connected/bias
VariableV2*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias
 
DDiscriminator/second/Discriminator/secondfully_connected/bias/AssignAssign=Discriminator/second/Discriminator/secondfully_connected/biasODiscriminator/second/Discriminator/secondfully_connected/bias/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:ђ
Ё
BDiscriminator/second/Discriminator/secondfully_connected/bias/readIdentity=Discriminator/second/Discriminator/secondfully_connected/bias*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
_output_shapes	
:ђ
Џ
?Discriminator/second/Discriminator/secondfully_connected/MatMulMatMul1Discriminator/first/Discriminator/firstleaky_reluDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b( *
T0
џ
@Discriminator/second/Discriminator/secondfully_connected/BiasAddBiasAdd?Discriminator/second/Discriminator/secondfully_connected/MatMulBDiscriminator/second/Discriminator/secondfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         ђ
~
9Discriminator/second/Discriminator/secondleaky_relu/alphaConst*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
Ь
7Discriminator/second/Discriminator/secondleaky_relu/mulMul9Discriminator/second/Discriminator/secondleaky_relu/alpha@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*(
_output_shapes
:         ђ*
T0
В
3Discriminator/second/Discriminator/secondleaky_reluMaximum7Discriminator/second/Discriminator/secondleaky_relu/mul@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
и
9Discriminator/out/kernel/Initializer/random_uniform/shapeConst*+
_class!
loc:@Discriminator/out/kernel*
valueB"      *
dtype0*
_output_shapes
:
Е
7Discriminator/out/kernel/Initializer/random_uniform/minConst*+
_class!
loc:@Discriminator/out/kernel*
valueB
 *IvЙ*
dtype0*
_output_shapes
: 
Е
7Discriminator/out/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *+
_class!
loc:@Discriminator/out/kernel*
valueB
 *Iv>
І
ADiscriminator/out/kernel/Initializer/random_uniform/RandomUniformRandomUniform9Discriminator/out/kernel/Initializer/random_uniform/shape*
_output_shapes
:	ђ*

seed*
T0*+
_class!
loc:@Discriminator/out/kernel*
seed2┌*
dtype0
■
7Discriminator/out/kernel/Initializer/random_uniform/subSub7Discriminator/out/kernel/Initializer/random_uniform/max7Discriminator/out/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
: 
Љ
7Discriminator/out/kernel/Initializer/random_uniform/mulMulADiscriminator/out/kernel/Initializer/random_uniform/RandomUniform7Discriminator/out/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
:	ђ
Ѓ
3Discriminator/out/kernel/Initializer/random_uniformAdd7Discriminator/out/kernel/Initializer/random_uniform/mul7Discriminator/out/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
:	ђ
╗
Discriminator/out/kernel
VariableV2*
dtype0*
_output_shapes
:	ђ*
shared_name *+
_class!
loc:@Discriminator/out/kernel*
	container *
shape:	ђ
Э
Discriminator/out/kernel/AssignAssignDiscriminator/out/kernel3Discriminator/out/kernel/Initializer/random_uniform*
T0*+
_class!
loc:@Discriminator/out/kernel*
validate_shape(*
_output_shapes
:	ђ*
use_locking(
џ
Discriminator/out/kernel/readIdentityDiscriminator/out/kernel*
T0*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
:	ђ
а
(Discriminator/out/bias/Initializer/zerosConst*
_output_shapes
:*)
_class
loc:@Discriminator/out/bias*
valueB*    *
dtype0
Г
Discriminator/out/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *)
_class
loc:@Discriminator/out/bias
Р
Discriminator/out/bias/AssignAssignDiscriminator/out/bias(Discriminator/out/bias/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@Discriminator/out/bias*
validate_shape(*
_output_shapes
:
Ј
Discriminator/out/bias/readIdentityDiscriminator/out/bias*
_output_shapes
:*
T0*)
_class
loc:@Discriminator/out/bias
╬
Discriminator/out/MatMulMatMul3Discriminator/second/Discriminator/secondleaky_reluDiscriminator/out/kernel/read*
transpose_b( *
T0*'
_output_shapes
:         *
transpose_a( 
ц
Discriminator/out/BiasAddBiasAddDiscriminator/out/MatMulDiscriminator/out/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
q
Discriminator/out/SigmoidSigmoidDiscriminator/out/BiasAdd*
T0*'
_output_shapes
:         
Ш
?Discriminator/first_1/Discriminator/firstfully_connected/MatMulMatMulGenerator/TanhBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b( *
T0
ў
@Discriminator/first_1/Discriminator/firstfully_connected/BiasAddBiasAdd?Discriminator/first_1/Discriminator/firstfully_connected/MatMul@Discriminator/first/Discriminator/firstfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         ђ
~
9Discriminator/first_1/Discriminator/firstleaky_relu/alphaConst*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
Ь
7Discriminator/first_1/Discriminator/firstleaky_relu/mulMul9Discriminator/first_1/Discriminator/firstleaky_relu/alpha@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
В
3Discriminator/first_1/Discriminator/firstleaky_reluMaximum7Discriminator/first_1/Discriminator/firstleaky_relu/mul@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
Ъ
ADiscriminator/second_1/Discriminator/secondfully_connected/MatMulMatMul3Discriminator/first_1/Discriminator/firstleaky_reluDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b( 
ъ
BDiscriminator/second_1/Discriminator/secondfully_connected/BiasAddBiasAddADiscriminator/second_1/Discriminator/secondfully_connected/MatMulBDiscriminator/second/Discriminator/secondfully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:         ђ*
T0
ђ
;Discriminator/second_1/Discriminator/secondleaky_relu/alphaConst*
_output_shapes
: *
valueB
 *═╠L>*
dtype0
З
9Discriminator/second_1/Discriminator/secondleaky_relu/mulMul;Discriminator/second_1/Discriminator/secondleaky_relu/alphaBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
Ы
5Discriminator/second_1/Discriminator/secondleaky_reluMaximum9Discriminator/second_1/Discriminator/secondleaky_relu/mulBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
м
Discriminator/out_1/MatMulMatMul5Discriminator/second_1/Discriminator/secondleaky_reluDiscriminator/out/kernel/read*
transpose_b( *
T0*'
_output_shapes
:         *
transpose_a( 
е
Discriminator/out_1/BiasAddBiasAddDiscriminator/out_1/MatMulDiscriminator/out/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
u
Discriminator/out_1/SigmoidSigmoidDiscriminator/out_1/BiasAdd*
T0*'
_output_shapes
:         
W
LogLogDiscriminator/out/Sigmoid*'
_output_shapes
:         *
T0
J
sub/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
`
subSubsub/xDiscriminator/out_1/Sigmoid*'
_output_shapes
:         *
T0
C
Log_1Logsub*
T0*'
_output_shapes
:         
H
addAddLogLog_1*
T0*'
_output_shapes
:         
V
ConstConst*
valueB"       *
dtype0*
_output_shapes
:
V
MeanMeanaddConst*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
1
NegNegMean*
_output_shapes
: *
T0
i
discriminator_loss/tagConst*
_output_shapes
: *#
valueB Bdiscriminator_loss*
dtype0
d
discriminator_lossHistogramSummarydiscriminator_loss/tagNeg*
T0*
_output_shapes
: 
L
sub_1/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
d
sub_1Subsub_1/xDiscriminator/out_1/Sigmoid*
T0*'
_output_shapes
:         
E
Log_2Logsub_1*'
_output_shapes
:         *
T0
X
Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
\
Mean_1MeanLog_2Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
a
generator_loss/tagConst*
valueB Bgenerator_loss*
dtype0*
_output_shapes
: 
_
generator_lossHistogramSummarygenerator_loss/tagMean_1*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
_output_shapes
: *
valueB *
dtype0
X
gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ђ?
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
N
gradients/Neg_grad/NegNeggradients/Fill*
T0*
_output_shapes
: 
r
!gradients/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
ў
gradients/Mean_grad/ReshapeReshapegradients/Neg_grad/Neg!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
\
gradients/Mean_grad/ShapeShapeadd*
T0*
out_type0*
_output_shapes
:
ю
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*'
_output_shapes
:         *

Tmultiples0
^
gradients/Mean_grad/Shape_1Shapeadd*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
ќ
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
џ
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
ѓ
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0
ђ
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
ї
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*'
_output_shapes
:         
[
gradients/add_grad/ShapeShapeLog*
_output_shapes
:*
T0*
out_type0
_
gradients/add_grad/Shape_1ShapeLog_1*
T0*
out_type0*
_output_shapes
:
┤
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
ц
gradients/add_grad/SumSumgradients/Mean_grad/truediv(gradients/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ќ
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
е
gradients/add_grad/Sum_1Sumgradients/Mean_grad/truediv*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ю
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*'
_output_shapes
:         *
T0*
Tshape0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
┌
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*'
_output_shapes
:         
Я
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*'
_output_shapes
:         
д
gradients/Log_grad/Reciprocal
ReciprocalDiscriminator/out/Sigmoid,^gradients/add_grad/tuple/control_dependency*'
_output_shapes
:         *
T0
Џ
gradients/Log_grad/mulMul+gradients/add_grad/tuple/control_dependencygradients/Log_grad/Reciprocal*
T0*'
_output_shapes
:         
ћ
gradients/Log_1_grad/Reciprocal
Reciprocalsub.^gradients/add_grad/tuple/control_dependency_1*'
_output_shapes
:         *
T0
А
gradients/Log_1_grad/mulMul-gradients/add_grad/tuple/control_dependency_1gradients/Log_1_grad/Reciprocal*'
_output_shapes
:         *
T0
е
4gradients/Discriminator/out/Sigmoid_grad/SigmoidGradSigmoidGradDiscriminator/out/Sigmoidgradients/Log_grad/mul*'
_output_shapes
:         *
T0
[
gradients/sub_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
u
gradients/sub_grad/Shape_1ShapeDiscriminator/out_1/Sigmoid*
T0*
out_type0*
_output_shapes
:
┤
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:         :         
А
gradients/sub_grad/SumSumgradients/Log_1_grad/mul(gradients/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
є
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
_output_shapes
: *
T0*
Tshape0
Ц
gradients/sub_grad/Sum_1Sumgradients/Log_1_grad/mul*gradients/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
T0*
_output_shapes
:
Џ
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*'
_output_shapes
:         *
T0*
Tshape0
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
╔
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*
_output_shapes
: 
Я
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*'
_output_shapes
:         *
T0
х
4gradients/Discriminator/out/BiasAdd_grad/BiasAddGradBiasAddGrad4gradients/Discriminator/out/Sigmoid_grad/SigmoidGrad*
data_formatNHWC*
_output_shapes
:*
T0
»
9gradients/Discriminator/out/BiasAdd_grad/tuple/group_depsNoOp5^gradients/Discriminator/out/BiasAdd_grad/BiasAddGrad5^gradients/Discriminator/out/Sigmoid_grad/SigmoidGrad
║
Agradients/Discriminator/out/BiasAdd_grad/tuple/control_dependencyIdentity4gradients/Discriminator/out/Sigmoid_grad/SigmoidGrad:^gradients/Discriminator/out/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:         *
T0*G
_class=
;9loc:@gradients/Discriminator/out/Sigmoid_grad/SigmoidGrad
»
Cgradients/Discriminator/out/BiasAdd_grad/tuple/control_dependency_1Identity4gradients/Discriminator/out/BiasAdd_grad/BiasAddGrad:^gradients/Discriminator/out/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/Discriminator/out/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
├
6gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGradSigmoidGradDiscriminator/out_1/Sigmoid-gradients/sub_grad/tuple/control_dependency_1*'
_output_shapes
:         *
T0
з
.gradients/Discriminator/out/MatMul_grad/MatMulMatMulAgradients/Discriminator/out/BiasAdd_grad/tuple/control_dependencyDiscriminator/out/kernel/read*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b(*
T0
ѓ
0gradients/Discriminator/out/MatMul_grad/MatMul_1MatMul3Discriminator/second/Discriminator/secondleaky_reluAgradients/Discriminator/out/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	ђ*
transpose_a(*
transpose_b( *
T0
ц
8gradients/Discriminator/out/MatMul_grad/tuple/group_depsNoOp/^gradients/Discriminator/out/MatMul_grad/MatMul1^gradients/Discriminator/out/MatMul_grad/MatMul_1
Г
@gradients/Discriminator/out/MatMul_grad/tuple/control_dependencyIdentity.gradients/Discriminator/out/MatMul_grad/MatMul9^gradients/Discriminator/out/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/Discriminator/out/MatMul_grad/MatMul*(
_output_shapes
:         ђ
ф
Bgradients/Discriminator/out/MatMul_grad/tuple/control_dependency_1Identity0gradients/Discriminator/out/MatMul_grad/MatMul_19^gradients/Discriminator/out/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/Discriminator/out/MatMul_grad/MatMul_1*
_output_shapes
:	ђ
╣
6gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGrad*
data_formatNHWC*
_output_shapes
:*
T0
х
;gradients/Discriminator/out_1/BiasAdd_grad/tuple/group_depsNoOp7^gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGrad7^gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGrad
┬
Cgradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGrad<^gradients/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGrad*'
_output_shapes
:         
и
Egradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency_1Identity6gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGrad<^gradients/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*I
_class?
=;loc:@gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGrad
┐
Hgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ShapeShape7Discriminator/second/Discriminator/secondleaky_relu/mul*
_output_shapes
:*
T0*
out_type0
╩
Jgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_1Shape@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
╩
Jgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_2Shape@gradients/Discriminator/out/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
Њ
Ngradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
▒
Hgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zerosFillJgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_2Ngradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:         ђ
Ї
Ogradients/Discriminator/second/Discriminator/secondleaky_relu_grad/GreaterEqualGreaterEqual7Discriminator/second/Discriminator/secondleaky_relu/mul@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
─
Xgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ShapeJgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:         :         
с
Igradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SelectSelectOgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/GreaterEqual@gradients/Discriminator/out/MatMul_grad/tuple/control_dependencyHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zeros*(
_output_shapes
:         ђ*
T0
т
Kgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Select_1SelectOgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/GreaterEqualHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zeros@gradients/Discriminator/out/MatMul_grad/tuple/control_dependency*(
_output_shapes
:         ђ*
T0
▓
Fgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SumSumIgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SelectXgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
е
Jgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ReshapeReshapeFgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SumHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape*
Tshape0*(
_output_shapes
:         ђ*
T0
И
Hgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Sum_1SumKgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Select_1Zgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
«
Lgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1ReshapeHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Sum_1Jgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:         ђ
э
Sgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/group_depsNoOpK^gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ReshapeM^gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1
Џ
[gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependencyIdentityJgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ReshapeT^gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/group_deps*]
_classS
QOloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape*(
_output_shapes
:         ђ*
T0
А
]gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1IdentityLgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1T^gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ
э
0gradients/Discriminator/out_1/MatMul_grad/MatMulMatMulCgradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/out/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b(
ѕ
2gradients/Discriminator/out_1/MatMul_grad/MatMul_1MatMul5Discriminator/second_1/Discriminator/secondleaky_reluCgradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	ђ*
transpose_a(*
transpose_b( 
ф
:gradients/Discriminator/out_1/MatMul_grad/tuple/group_depsNoOp1^gradients/Discriminator/out_1/MatMul_grad/MatMul3^gradients/Discriminator/out_1/MatMul_grad/MatMul_1
х
Bgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependencyIdentity0gradients/Discriminator/out_1/MatMul_grad/MatMul;^gradients/Discriminator/out_1/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/Discriminator/out_1/MatMul_grad/MatMul*(
_output_shapes
:         ђ
▓
Dgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency_1Identity2gradients/Discriminator/out_1/MatMul_grad/MatMul_1;^gradients/Discriminator/out_1/MatMul_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/Discriminator/out_1/MatMul_grad/MatMul_1*
_output_shapes
:	ђ
Ў
gradients/AddNAddNCgradients/Discriminator/out/BiasAdd_grad/tuple/control_dependency_1Egradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency_1*
N*
_output_shapes
:*
T0*G
_class=
;9loc:@gradients/Discriminator/out/BiasAdd_grad/BiasAddGrad
Ј
Lgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
╬
Ngradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape_1Shape@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
л
\gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ShapeNgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Б
Jgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/MulMul[gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*(
_output_shapes
:         ђ*
T0
╗
Jgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/SumSumJgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Mul\gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
б
Ngradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ReshapeReshapeJgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/SumLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape*
Tshape0*
_output_shapes
: *
T0
ъ
Lgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Mul_1Mul9Discriminator/second/Discriminator/secondleaky_relu/alpha[gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:         ђ
┴
Lgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Sum_1SumLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Mul_1^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
║
Pgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape_1ReshapeLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Sum_1Ngradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:         ђ
Ѓ
Wgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/group_depsNoOpO^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ReshapeQ^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape_1
Ў
_gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependencyIdentityNgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ReshapeX^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape*
_output_shapes
: 
▒
agradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1IdentityPgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape_1X^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:         ђ
├
Jgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeShape9Discriminator/second_1/Discriminator/secondleaky_relu/mul*
_output_shapes
:*
T0*
out_type0
╬
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1ShapeBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
╬
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_2ShapeBgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency*
out_type0*
_output_shapes
:*
T0
Ћ
Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
и
Jgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zerosFillLgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_2Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:         ђ
Њ
Qgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualGreaterEqual9Discriminator/second_1/Discriminator/secondleaky_relu/mulBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
╩
Zgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeLgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1*2
_output_shapes 
:         :         *
T0
в
Kgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SelectSelectQgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualBgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependencyJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros*(
_output_shapes
:         ђ*
T0
ь
Mgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1SelectQgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zerosBgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency*(
_output_shapes
:         ђ*
T0
И
Hgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SumSumKgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SelectZgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
«
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeReshapeHgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SumJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         ђ
Й
Jgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1SumMgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1\gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
┤
Ngradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1ReshapeJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:         ђ
§
Ugradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_depsNoOpM^gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeO^gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1
Б
]gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependencyIdentityLgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeV^gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape*(
_output_shapes
:         ђ
Е
_gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1IdentityNgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1V^gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ
џ
gradients/AddN_1AddNBgradients/Discriminator/out/MatMul_grad/tuple/control_dependency_1Dgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency_1*
_output_shapes
:	ђ*
T0*C
_class9
75loc:@gradients/Discriminator/out/MatMul_grad/MatMul_1*
N
э
gradients/AddN_2AddN]gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1agradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:         ђ*
T0*_
_classU
SQloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1
╣
[gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_2*
data_formatNHWC*
_output_shapes	
:ђ*
T0
┘
`gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_2\^gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad
§
hgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_2a^gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ
╠
jgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity[gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrada^gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*n
_classd
b`loc:@gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:ђ
Љ
Ngradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
м
Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1ShapeBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
о
^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ShapePgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Е
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/MulMul]gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependencyBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*(
_output_shapes
:         ђ*
T0
┴
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumSumLgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
е
Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeReshapeLgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumNgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape*
Tshape0*
_output_shapes
: *
T0
ц
Ngradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul_1Mul;Discriminator/second_1/Discriminator/secondleaky_relu/alpha]gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency*(
_output_shapes
:         ђ*
T0
К
Ngradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1SumNgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul_1`gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
└
Rgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1ReshapeNgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1*(
_output_shapes
:         ђ*
T0*
Tshape0
Ѕ
Ygradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_depsNoOpQ^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeS^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1
А
agradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependencyIdentityPgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeZ^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape*
_output_shapes
: 
╣
cgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1IdentityRgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1Z^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:         ђ
У
Ugradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMulMatMulhgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*
transpose_b(*
T0*(
_output_shapes
:         ђ*
transpose_a( 
¤
Wgradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1MatMul1Discriminator/first/Discriminator/firstleaky_reluhgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
ђђ*
transpose_a(*
transpose_b( *
T0
Ў
_gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/group_depsNoOpV^gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMulX^gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1
╔
ggradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyIdentityUgradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul`^gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul*(
_output_shapes
:         ђ
К
igradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1IdentityWgradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1`^gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*j
_class`
^\loc:@gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
ђђ
§
gradients/AddN_3AddN_gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1cgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1*
T0*a
_classW
USloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:         ђ
╗
]gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_3*
data_formatNHWC*
_output_shapes	
:ђ*
T0
П
bgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_3^^gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad
Ѓ
jgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_3c^gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ
н
lgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity]gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGradc^gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:ђ*
T0*p
_classf
dbloc:@gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad
╗
Fgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ShapeShape5Discriminator/first/Discriminator/firstleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
к
Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_1Shape>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
№
Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_2Shapeggradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
out_type0*
_output_shapes
:*
T0
Љ
Lgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ф
Fgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zerosFillHgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_2Lgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:         ђ
Є
Mgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/GreaterEqualGreaterEqual5Discriminator/first/Discriminator/firstleaky_relu/mul>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*(
_output_shapes
:         ђ*
T0
Й
Vgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ShapeHgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:         :         
ё
Ggradients/Discriminator/first/Discriminator/firstleaky_relu_grad/SelectSelectMgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/GreaterEqualggradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zeros*(
_output_shapes
:         ђ*
T0
є
Igradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Select_1SelectMgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/GreaterEqualFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zerosggradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:         ђ*
T0
г
Dgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/SumSumGgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/SelectVgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
б
Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ReshapeReshapeDgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/SumFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         ђ
▓
Fgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Sum_1SumIgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Select_1Xgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
е
Jgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1ReshapeFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Sum_1Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:         ђ
ы
Qgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/group_depsNoOpI^gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ReshapeK^gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1
Њ
Ygradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependencyIdentityHgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ReshapeR^gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/group_deps*(
_output_shapes
:         ђ*
T0*[
_classQ
OMloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape
Ў
[gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1IdentityJgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1R^gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ
В
Wgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMulMatMuljgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b(
Н
Ygradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1MatMul3Discriminator/first_1/Discriminator/firstleaky_relujgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
ђђ*
transpose_a(*
transpose_b( *
T0
Ъ
agradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_depsNoOpX^gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMulZ^gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1
Л
igradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyIdentityWgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMulb^gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*j
_class`
^\loc:@gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul*(
_output_shapes
:         ђ
¤
kgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1IdentityYgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1b^gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*l
_classb
`^loc:@gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
ђђ
Љ
gradients/AddN_4AddNjgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1lgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1*n
_classd
b`loc:@gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:ђ*
T0
Ї
Jgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
╩
Lgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape_1Shape>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
╩
Zgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ShapeLgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Ю
Hgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/MulMulYgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependency>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
х
Hgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/SumSumHgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/MulZgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ю
Lgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ReshapeReshapeHgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/SumJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
ў
Jgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Mul_1Mul7Discriminator/first/Discriminator/firstleaky_relu/alphaYgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:         ђ
╗
Jgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Sum_1SumJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Mul_1\gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
┤
Ngradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1ReshapeJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Sum_1Lgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:         ђ
§
Ugradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/group_depsNoOpM^gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ReshapeO^gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1
Љ
]gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependencyIdentityLgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ReshapeV^gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*_
_classU
SQloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape
Е
_gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityNgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1V^gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:         ђ
┐
Hgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ShapeShape7Discriminator/first_1/Discriminator/firstleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
╩
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1Shape@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
з
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_2Shapeigradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
Њ
Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
▒
Hgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zerosFillJgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_2Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:         ђ
Ї
Ogradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualGreaterEqual7Discriminator/first_1/Discriminator/firstleaky_relu/mul@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*(
_output_shapes
:         ђ*
T0
─
Xgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ShapeJgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1*2
_output_shapes 
:         :         *
T0
ї
Igradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SelectSelectOgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualigradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros*(
_output_shapes
:         ђ*
T0
ј
Kgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1SelectOgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zerosigradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:         ђ
▓
Fgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumSumIgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SelectXgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
е
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeReshapeFgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape*(
_output_shapes
:         ђ*
T0*
Tshape0
И
Hgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Sum_1SumKgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1Zgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
«
Lgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1ReshapeHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Sum_1Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:         ђ
э
Sgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_depsNoOpK^gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeM^gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1
Џ
[gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependencyIdentityJgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeT^gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape*(
_output_shapes
:         ђ
А
]gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1IdentityLgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1T^gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ
љ
gradients/AddN_5AddNigradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1kgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1*j
_class`
^\loc:@gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1*
N* 
_output_shapes
:
ђђ*
T0
ы
gradients/AddN_6AddN[gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1_gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1*
T0*]
_classS
QOloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:         ђ
и
Ygradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_6*
T0*
data_formatNHWC*
_output_shapes	
:ђ
Н
^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_6Z^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad
э
fgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_6_^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ
─
hgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityYgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad_^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*l
_classb
`^loc:@gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:ђ
Ј
Lgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
╬
Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1Shape@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
л
\gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ShapeNgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Б
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/MulMul[gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
╗
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumSumJgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul\gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
б
Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeReshapeJgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumLgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ъ
Lgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul_1Mul9Discriminator/first_1/Discriminator/firstleaky_relu/alpha[gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency*(
_output_shapes
:         ђ*
T0
┴
Lgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Sum_1SumLgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul_1^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
║
Pgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1ReshapeLgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Sum_1Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:         ђ
Ѓ
Wgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_depsNoOpO^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeQ^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1
Ў
_gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependencyIdentityNgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeX^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*a
_classW
USloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape*
_output_shapes
: *
T0
▒
agradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityPgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1X^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:         ђ
Р
Sgradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMulMatMulfgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
T0*(
_output_shapes
:         љ*
transpose_a( *
transpose_b(
г
Ugradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1MatMulDiscriminator/realfgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
љђ*
transpose_a(*
transpose_b( 
Њ
]gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/group_depsNoOpT^gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMulV^gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
┴
egradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentitySgradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul^^gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul*(
_output_shapes
:         љ
┐
ggradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityUgradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1^^gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
љђ
э
gradients/AddN_7AddN]gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1agradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1*
T0*_
_classU
SQloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:         ђ
╣
[gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_7*
_output_shapes	
:ђ*
T0*
data_formatNHWC
┘
`gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_7\^gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad
§
hgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_7a^gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ
╠
jgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity[gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrada^gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*n
_classd
b`loc:@gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:ђ
Т
Ugradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulMatMulhgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
T0*(
_output_shapes
:         љ*
transpose_a( *
transpose_b(
г
Wgradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1MatMulGenerator/Tanhhgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
љђ*
transpose_a(
Ў
_gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_depsNoOpV^gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulX^gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
╔
ggradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentityUgradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul`^gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul*(
_output_shapes
:         љ
К
igradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityWgradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1`^gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*j
_class`
^\loc:@gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
љђ
І
gradients/AddN_8AddNhgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1jgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*l
_classb
`^loc:@gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:ђ
і
gradients/AddN_9AddNggradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1igradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1*
T0*h
_class^
\Zloc:@gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1*
N* 
_output_shapes
:
љђ
«
beta1_power/initial_valueConst*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
┐
beta1_power
VariableV2*
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
я
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes
: 
џ
beta1_power/readIdentitybeta1_power*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes
: 
«
beta2_power/initial_valueConst*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueB
 *wЙ?*
dtype0*
_output_shapes
: 
┐
beta2_power
VariableV2*
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
я
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias
џ
beta2_power/readIdentitybeta2_power*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes
: 
Є
dDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
ы
ZDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/ConstConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Г
TDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zerosFilldDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorZDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
љђ*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*

index_type0
ї
BDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam
VariableV2*
	container *
shape:
љђ*
dtype0* 
_output_shapes
:
љђ*
shared_name *P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel
Њ
IDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/AssignAssignBDiscriminator/first/Discriminator/firstfully_connected/kernel/AdamTDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
validate_shape(* 
_output_shapes
:
љђ*
use_locking(
ћ
GDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/readIdentityBDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel* 
_output_shapes
:
љђ
Ѕ
fDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
з
\Discriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
│
VDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zerosFillfDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensor\Discriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*

index_type0* 
_output_shapes
:
љђ
ј
DDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
љђ*
shared_name *P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
	container *
shape:
љђ
Ў
KDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/AssignAssignDDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1VDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
validate_shape(* 
_output_shapes
:
љђ
ў
IDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/readIdentityDDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel* 
_output_shapes
:
љђ*
T0
ы
RDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/Initializer/zerosConst*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
■
@Discriminator/first/Discriminator/firstfully_connected/bias/Adam
VariableV2*
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ
є
GDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/AssignAssign@Discriminator/first/Discriminator/firstfully_connected/bias/AdamRDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:ђ
Ѕ
EDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/readIdentity@Discriminator/first/Discriminator/firstfully_connected/bias/Adam*
_output_shapes	
:ђ*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias
з
TDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:ђ*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueBђ*    *
dtype0
ђ
BDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1
VariableV2*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name 
ї
IDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/AssignAssignBDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1TDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:ђ
Ї
GDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/readIdentityBDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes	
:ђ
І
fDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
ш
\Discriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB
 *    
х
VDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zerosFillfDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensor\Discriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros/Const*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*

index_type0* 
_output_shapes
:
ђђ
љ
DDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam
VariableV2*
	container *
shape:
ђђ*
dtype0* 
_output_shapes
:
ђђ*
shared_name *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel
Џ
KDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/AssignAssignDDiscriminator/second/Discriminator/secondfully_connected/kernel/AdamVDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel
џ
IDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/readIdentityDDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam* 
_output_shapes
:
ђђ*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel
Ї
hDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
э
^Discriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
╗
XDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zerosFillhDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensor^Discriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
ђђ*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*

index_type0
њ
FDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1
VariableV2*
	container *
shape:
ђђ*
dtype0* 
_output_shapes
:
ђђ*
shared_name *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel
А
MDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/AssignAssignFDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1XDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:
ђђ
ъ
KDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/readIdentityFDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel* 
_output_shapes
:
ђђ*
T0
ш
TDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/Initializer/zerosConst*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
ѓ
BDiscriminator/second/Discriminator/secondfully_connected/bias/Adam
VariableV2*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name 
ј
IDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/AssignAssignBDiscriminator/second/Discriminator/secondfully_connected/bias/AdamTDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/Initializer/zeros*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0
Ј
GDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/readIdentityBDiscriminator/second/Discriminator/secondfully_connected/bias/Adam*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
_output_shapes	
:ђ
э
VDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/Initializer/zerosConst*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
ё
DDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1
VariableV2*
_output_shapes	
:ђ*
shared_name *P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
	container *
shape:ђ*
dtype0
ћ
KDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/AssignAssignDDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1VDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:ђ
Њ
IDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/readIdentityDDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
_output_shapes	
:ђ
│
/Discriminator/out/kernel/Adam/Initializer/zerosConst*+
_class!
loc:@Discriminator/out/kernel*
valueB	ђ*    *
dtype0*
_output_shapes
:	ђ
└
Discriminator/out/kernel/Adam
VariableV2*+
_class!
loc:@Discriminator/out/kernel*
	container *
shape:	ђ*
dtype0*
_output_shapes
:	ђ*
shared_name 
■
$Discriminator/out/kernel/Adam/AssignAssignDiscriminator/out/kernel/Adam/Discriminator/out/kernel/Adam/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@Discriminator/out/kernel*
validate_shape(*
_output_shapes
:	ђ
ц
"Discriminator/out/kernel/Adam/readIdentityDiscriminator/out/kernel/Adam*
T0*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
:	ђ
х
1Discriminator/out/kernel/Adam_1/Initializer/zerosConst*+
_class!
loc:@Discriminator/out/kernel*
valueB	ђ*    *
dtype0*
_output_shapes
:	ђ
┬
Discriminator/out/kernel/Adam_1
VariableV2*
shared_name *+
_class!
loc:@Discriminator/out/kernel*
	container *
shape:	ђ*
dtype0*
_output_shapes
:	ђ
ё
&Discriminator/out/kernel/Adam_1/AssignAssignDiscriminator/out/kernel/Adam_11Discriminator/out/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@Discriminator/out/kernel*
validate_shape(*
_output_shapes
:	ђ
е
$Discriminator/out/kernel/Adam_1/readIdentityDiscriminator/out/kernel/Adam_1*
T0*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
:	ђ
Ц
-Discriminator/out/bias/Adam/Initializer/zerosConst*)
_class
loc:@Discriminator/out/bias*
valueB*    *
dtype0*
_output_shapes
:
▓
Discriminator/out/bias/Adam
VariableV2*
shared_name *)
_class
loc:@Discriminator/out/bias*
	container *
shape:*
dtype0*
_output_shapes
:
ы
"Discriminator/out/bias/Adam/AssignAssignDiscriminator/out/bias/Adam-Discriminator/out/bias/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@Discriminator/out/bias*
validate_shape(*
_output_shapes
:
Ў
 Discriminator/out/bias/Adam/readIdentityDiscriminator/out/bias/Adam*
_output_shapes
:*
T0*)
_class
loc:@Discriminator/out/bias
Д
/Discriminator/out/bias/Adam_1/Initializer/zerosConst*)
_class
loc:@Discriminator/out/bias*
valueB*    *
dtype0*
_output_shapes
:
┤
Discriminator/out/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *)
_class
loc:@Discriminator/out/bias*
	container *
shape:
э
$Discriminator/out/bias/Adam_1/AssignAssignDiscriminator/out/bias/Adam_1/Discriminator/out/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@Discriminator/out/bias*
validate_shape(*
_output_shapes
:
Ю
"Discriminator/out/bias/Adam_1/readIdentityDiscriminator/out/bias/Adam_1*
T0*)
_class
loc:@Discriminator/out/bias*
_output_shapes
:
W
Adam/learning_rateConst*
valueB
 *иQ9*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *wЙ?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
й
SAdam/update_Discriminator/first/Discriminator/firstfully_connected/kernel/ApplyAdam	ApplyAdam=Discriminator/first/Discriminator/firstfully_connected/kernelBDiscriminator/first/Discriminator/firstfully_connected/kernel/AdamDDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_9*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
use_nesterov( * 
_output_shapes
:
љђ*
use_locking( 
«
QAdam/update_Discriminator/first/Discriminator/firstfully_connected/bias/ApplyAdam	ApplyAdam;Discriminator/first/Discriminator/firstfully_connected/bias@Discriminator/first/Discriminator/firstfully_connected/bias/AdamBDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_8*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
use_nesterov( *
_output_shapes	
:ђ*
use_locking( *
T0
К
UAdam/update_Discriminator/second/Discriminator/secondfully_connected/kernel/ApplyAdam	ApplyAdam?Discriminator/second/Discriminator/secondfully_connected/kernelDDiscriminator/second/Discriminator/secondfully_connected/kernel/AdamFDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_5*
use_nesterov( * 
_output_shapes
:
ђђ*
use_locking( *
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel
И
SAdam/update_Discriminator/second/Discriminator/secondfully_connected/bias/ApplyAdam	ApplyAdam=Discriminator/second/Discriminator/secondfully_connected/biasBDiscriminator/second/Discriminator/secondfully_connected/bias/AdamDDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_4*
use_locking( *
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
use_nesterov( *
_output_shapes	
:ђ
Ѓ
.Adam/update_Discriminator/out/kernel/ApplyAdam	ApplyAdamDiscriminator/out/kernelDiscriminator/out/kernel/AdamDiscriminator/out/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_1*
use_locking( *
T0*+
_class!
loc:@Discriminator/out/kernel*
use_nesterov( *
_output_shapes
:	ђ
Ы
,Adam/update_Discriminator/out/bias/ApplyAdam	ApplyAdamDiscriminator/out/biasDiscriminator/out/bias/AdamDiscriminator/out/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN*
use_locking( *
T0*)
_class
loc:@Discriminator/out/bias*
use_nesterov( *
_output_shapes
:
о
Adam/mulMulbeta1_power/read
Adam/beta1R^Adam/update_Discriminator/first/Discriminator/firstfully_connected/bias/ApplyAdamT^Adam/update_Discriminator/first/Discriminator/firstfully_connected/kernel/ApplyAdam-^Adam/update_Discriminator/out/bias/ApplyAdam/^Adam/update_Discriminator/out/kernel/ApplyAdamT^Adam/update_Discriminator/second/Discriminator/secondfully_connected/bias/ApplyAdamV^Adam/update_Discriminator/second/Discriminator/secondfully_connected/kernel/ApplyAdam*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes
: *
T0
к
Adam/AssignAssignbeta1_powerAdam/mul*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes
: *
use_locking( 
п

Adam/mul_1Mulbeta2_power/read
Adam/beta2R^Adam/update_Discriminator/first/Discriminator/firstfully_connected/bias/ApplyAdamT^Adam/update_Discriminator/first/Discriminator/firstfully_connected/kernel/ApplyAdam-^Adam/update_Discriminator/out/bias/ApplyAdam/^Adam/update_Discriminator/out/kernel/ApplyAdamT^Adam/update_Discriminator/second/Discriminator/secondfully_connected/bias/ApplyAdamV^Adam/update_Discriminator/second/Discriminator/secondfully_connected/kernel/ApplyAdam*
_output_shapes
: *
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias
╩
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes
: 
Р
AdamNoOp^Adam/Assign^Adam/Assign_1R^Adam/update_Discriminator/first/Discriminator/firstfully_connected/bias/ApplyAdamT^Adam/update_Discriminator/first/Discriminator/firstfully_connected/kernel/ApplyAdam-^Adam/update_Discriminator/out/bias/ApplyAdam/^Adam/update_Discriminator/out/kernel/ApplyAdamT^Adam/update_Discriminator/second/Discriminator/secondfully_connected/bias/ApplyAdamV^Adam/update_Discriminator/second/Discriminator/secondfully_connected/kernel/ApplyAdam
T
gradients_1/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
gradients_1/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ђ?
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
_output_shapes
: *
T0*

index_type0
v
%gradients_1/Mean_1_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
џ
gradients_1/Mean_1_grad/ReshapeReshapegradients_1/Fill%gradients_1/Mean_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
b
gradients_1/Mean_1_grad/ShapeShapeLog_2*
T0*
out_type0*
_output_shapes
:
е
gradients_1/Mean_1_grad/TileTilegradients_1/Mean_1_grad/Reshapegradients_1/Mean_1_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:         
d
gradients_1/Mean_1_grad/Shape_1ShapeLog_2*
T0*
out_type0*
_output_shapes
:
b
gradients_1/Mean_1_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
g
gradients_1/Mean_1_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0
б
gradients_1/Mean_1_grad/ProdProdgradients_1/Mean_1_grad/Shape_1gradients_1/Mean_1_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
i
gradients_1/Mean_1_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
д
gradients_1/Mean_1_grad/Prod_1Prodgradients_1/Mean_1_grad/Shape_2gradients_1/Mean_1_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
c
!gradients_1/Mean_1_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
ј
gradients_1/Mean_1_grad/MaximumMaximumgradients_1/Mean_1_grad/Prod_1!gradients_1/Mean_1_grad/Maximum/y*
T0*
_output_shapes
: 
ї
 gradients_1/Mean_1_grad/floordivFloorDivgradients_1/Mean_1_grad/Prodgradients_1/Mean_1_grad/Maximum*
_output_shapes
: *
T0
є
gradients_1/Mean_1_grad/CastCast gradients_1/Mean_1_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
ў
gradients_1/Mean_1_grad/truedivRealDivgradients_1/Mean_1_grad/Tilegradients_1/Mean_1_grad/Cast*
T0*'
_output_shapes
:         
і
!gradients_1/Log_2_grad/Reciprocal
Reciprocalsub_1 ^gradients_1/Mean_1_grad/truediv*'
_output_shapes
:         *
T0
Ќ
gradients_1/Log_2_grad/mulMulgradients_1/Mean_1_grad/truediv!gradients_1/Log_2_grad/Reciprocal*
T0*'
_output_shapes
:         
_
gradients_1/sub_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
y
gradients_1/sub_1_grad/Shape_1ShapeDiscriminator/out_1/Sigmoid*
_output_shapes
:*
T0*
out_type0
└
,gradients_1/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_1_grad/Shapegradients_1/sub_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Ф
gradients_1/sub_1_grad/SumSumgradients_1/Log_2_grad/mul,gradients_1/sub_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
њ
gradients_1/sub_1_grad/ReshapeReshapegradients_1/sub_1_grad/Sumgradients_1/sub_1_grad/Shape*
_output_shapes
: *
T0*
Tshape0
»
gradients_1/sub_1_grad/Sum_1Sumgradients_1/Log_2_grad/mul.gradients_1/sub_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
b
gradients_1/sub_1_grad/NegNeggradients_1/sub_1_grad/Sum_1*
T0*
_output_shapes
:
Д
 gradients_1/sub_1_grad/Reshape_1Reshapegradients_1/sub_1_grad/Neggradients_1/sub_1_grad/Shape_1*
Tshape0*'
_output_shapes
:         *
T0
s
'gradients_1/sub_1_grad/tuple/group_depsNoOp^gradients_1/sub_1_grad/Reshape!^gradients_1/sub_1_grad/Reshape_1
┘
/gradients_1/sub_1_grad/tuple/control_dependencyIdentitygradients_1/sub_1_grad/Reshape(^gradients_1/sub_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/sub_1_grad/Reshape*
_output_shapes
: 
­
1gradients_1/sub_1_grad/tuple/control_dependency_1Identity gradients_1/sub_1_grad/Reshape_1(^gradients_1/sub_1_grad/tuple/group_deps*'
_output_shapes
:         *
T0*3
_class)
'%loc:@gradients_1/sub_1_grad/Reshape_1
╔
8gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGradSigmoidGradDiscriminator/out_1/Sigmoid1gradients_1/sub_1_grad/tuple/control_dependency_1*'
_output_shapes
:         *
T0
й
8gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGradBiasAddGrad8gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad*
T0*
data_formatNHWC*
_output_shapes
:
╗
=gradients_1/Discriminator/out_1/BiasAdd_grad/tuple/group_depsNoOp9^gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGrad9^gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad
╩
Egradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyIdentity8gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad>^gradients_1/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad*'
_output_shapes
:         
┐
Ggradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency_1Identity8gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGrad>^gradients_1/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*K
_classA
?=loc:@gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
ч
2gradients_1/Discriminator/out_1/MatMul_grad/MatMulMatMulEgradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/out/kernel/read*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b(*
T0
ї
4gradients_1/Discriminator/out_1/MatMul_grad/MatMul_1MatMul5Discriminator/second_1/Discriminator/secondleaky_reluEgradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	ђ*
transpose_a(*
transpose_b( 
░
<gradients_1/Discriminator/out_1/MatMul_grad/tuple/group_depsNoOp3^gradients_1/Discriminator/out_1/MatMul_grad/MatMul5^gradients_1/Discriminator/out_1/MatMul_grad/MatMul_1
й
Dgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependencyIdentity2gradients_1/Discriminator/out_1/MatMul_grad/MatMul=^gradients_1/Discriminator/out_1/MatMul_grad/tuple/group_deps*E
_class;
97loc:@gradients_1/Discriminator/out_1/MatMul_grad/MatMul*(
_output_shapes
:         ђ*
T0
║
Fgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependency_1Identity4gradients_1/Discriminator/out_1/MatMul_grad/MatMul_1=^gradients_1/Discriminator/out_1/MatMul_grad/tuple/group_deps*
_output_shapes
:	ђ*
T0*G
_class=
;9loc:@gradients_1/Discriminator/out_1/MatMul_grad/MatMul_1
┼
Lgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeShape9Discriminator/second_1/Discriminator/secondleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
л
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1ShapeBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
м
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_2ShapeDgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
Ќ
Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
й
Lgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zerosFillNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_2Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:         ђ
Ћ
Sgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualGreaterEqual9Discriminator/second_1/Discriminator/secondleaky_relu/mulBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
л
\gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1*2
_output_shapes 
:         :         *
T0
з
Mgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SelectSelectSgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualDgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependencyLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros*(
_output_shapes
:         ђ*
T0
ш
Ogradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1SelectSgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zerosDgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:         ђ
Й
Jgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SumSumMgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select\gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
┤
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeReshapeJgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SumLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         ђ
─
Lgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1SumOgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
║
Pgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1ReshapeLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:         ђ
Ѓ
Wgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_depsNoOpO^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeQ^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1
Ф
_gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependencyIdentityNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeX^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape*(
_output_shapes
:         ђ
▒
agradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1IdentityPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1X^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_deps*c
_classY
WUloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ*
T0
Њ
Pgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
н
Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1ShapeBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
▄
`gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ShapeRgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Г
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/MulMul_gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependencyBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
К
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumSumNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul`gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
«
Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeReshapeNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
е
Pgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul_1Mul;Discriminator/second_1/Discriminator/secondleaky_relu/alpha_gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency*(
_output_shapes
:         ђ*
T0
═
Pgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1SumPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul_1bgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
к
Tgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1ReshapePgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:         ђ
Ј
[gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_depsNoOpS^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeU^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1
Е
cgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependencyIdentityRgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape\^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*e
_class[
YWloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape
┴
egradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1IdentityTgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1\^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*g
_class]
[Yloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:         ђ*
T0
Ѓ
gradients_1/AddNAddNagradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1egradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1*
T0*c
_classY
WUloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:         ђ
й
_gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN*
_output_shapes	
:ђ*
T0*
data_formatNHWC
р
dgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN`^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad
Ѕ
lgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddNe^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ
▄
ngradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity_gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrade^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*r
_classh
fdloc:@gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:ђ
­
Ygradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMulMatMullgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*
T0*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b(
┘
[gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1MatMul3Discriminator/first_1/Discriminator/firstleaky_relulgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
ђђ*
transpose_a(
Ц
cgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_depsNoOpZ^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul\^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1
┘
kgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyIdentityYgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMuld^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*l
_classb
`^loc:@gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul*(
_output_shapes
:         ђ
О
mgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1Identity[gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1d^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*n
_classd
b`loc:@gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
ђђ
┴
Jgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ShapeShape7Discriminator/first_1/Discriminator/firstleaky_relu/mul*
_output_shapes
:*
T0*
out_type0
╠
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1Shape@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
э
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_2Shapekgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
Ћ
Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
и
Jgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zerosFillLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_2Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:         ђ
Ј
Qgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualGreaterEqual7Discriminator/first_1/Discriminator/firstleaky_relu/mul@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
╩
Zgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ShapeLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:         :         
ћ
Kgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SelectSelectQgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualkgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros*(
_output_shapes
:         ђ*
T0
ќ
Mgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1SelectQgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeroskgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:         ђ*
T0
И
Hgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumSumKgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SelectZgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
«
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeReshapeHgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         ђ
Й
Jgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Sum_1SumMgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1\gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
┤
Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1ReshapeJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Sum_1Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1*
Tshape0*(
_output_shapes
:         ђ*
T0
§
Ugradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_depsNoOpM^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeO^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1
Б
]gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependencyIdentityLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeV^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_deps*(
_output_shapes
:         ђ*
T0*_
_classU
SQloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape
Е
_gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1IdentityNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1V^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_deps*(
_output_shapes
:         ђ*
T0*a
_classW
USloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1
Љ
Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
л
Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1Shape@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
о
^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ShapePgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Д
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/MulMul]gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
┴
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumSumLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
е
Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeReshapeLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
б
Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul_1Mul9Discriminator/first_1/Discriminator/firstleaky_relu/alpha]gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:         ђ
К
Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Sum_1SumNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul_1`gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
└
Rgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1ReshapeNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Sum_1Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:         ђ
Ѕ
Ygradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_depsNoOpQ^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeS^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1
А
agradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependencyIdentityPgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeZ^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape*
_output_shapes
: 
╣
cgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityRgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1Z^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:         ђ
 
gradients_1/AddN_1AddN_gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1cgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1*(
_output_shapes
:         ђ*
T0*a
_classW
USloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*
N
й
]gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_1*
T0*
data_formatNHWC*
_output_shapes	
:ђ
▀
bgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_1^^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad
Ё
jgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_1c^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ
н
lgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity]gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradc^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*p
_classf
dbloc:@gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:ђ
Ж
Wgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulMatMuljgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
T0*(
_output_shapes
:         љ*
transpose_a( *
transpose_b(
░
Ygradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1MatMulGenerator/Tanhjgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
љђ*
transpose_a(*
transpose_b( 
Ъ
agradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_depsNoOpX^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulZ^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
Л
igradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentityWgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulb^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*j
_class`
^\loc:@gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul*(
_output_shapes
:         љ
¤
kgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityYgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1b^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
љђ*
T0*l
_classb
`^loc:@gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
Р
(gradients_1/Generator/Tanh_grad/TanhGradTanhGradGenerator/Tanhigradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:         љ
ф
4gradients_1/Generator/dense/BiasAdd_grad/BiasAddGradBiasAddGrad(gradients_1/Generator/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes	
:љ
Б
9gradients_1/Generator/dense/BiasAdd_grad/tuple/group_depsNoOp)^gradients_1/Generator/Tanh_grad/TanhGrad5^gradients_1/Generator/dense/BiasAdd_grad/BiasAddGrad
Б
Agradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependencyIdentity(gradients_1/Generator/Tanh_grad/TanhGrad:^gradients_1/Generator/dense/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/Generator/Tanh_grad/TanhGrad*(
_output_shapes
:         љ
░
Cgradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependency_1Identity4gradients_1/Generator/dense/BiasAdd_grad/BiasAddGrad:^gradients_1/Generator/dense/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/Generator/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:љ
ы
.gradients_1/Generator/dense/MatMul_grad/MatMulMatMulAgradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependencyGenerator/dense/kernel/read*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b(*
T0
щ
0gradients_1/Generator/dense/MatMul_grad/MatMul_1MatMul)Generator/forth/Generator/forthleaky_reluAgradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
ђљ*
transpose_a(
ц
8gradients_1/Generator/dense/MatMul_grad/tuple/group_depsNoOp/^gradients_1/Generator/dense/MatMul_grad/MatMul1^gradients_1/Generator/dense/MatMul_grad/MatMul_1
Г
@gradients_1/Generator/dense/MatMul_grad/tuple/control_dependencyIdentity.gradients_1/Generator/dense/MatMul_grad/MatMul9^gradients_1/Generator/dense/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/Generator/dense/MatMul_grad/MatMul*(
_output_shapes
:         ђ
Ф
Bgradients_1/Generator/dense/MatMul_grad/tuple/control_dependency_1Identity0gradients_1/Generator/dense/MatMul_grad/MatMul_19^gradients_1/Generator/dense/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/Generator/dense/MatMul_grad/MatMul_1* 
_output_shapes
:
ђљ
Г
@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/ShapeShape-Generator/forth/Generator/forthleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
┴
Bgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_1Shape?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
┬
Bgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_2Shape@gradients_1/Generator/dense/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
І
Fgradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ў
@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/zerosFillBgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_2Fgradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:         ђ
Щ
Ggradients_1/Generator/forth/Generator/forthleaky_relu_grad/GreaterEqualGreaterEqual-Generator/forth/Generator/forthleaky_relu/mul?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*(
_output_shapes
:         ђ*
T0
г
Pgradients_1/Generator/forth/Generator/forthleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/ShapeBgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_1*2
_output_shapes 
:         :         *
T0
╦
Agradients_1/Generator/forth/Generator/forthleaky_relu_grad/SelectSelectGgradients_1/Generator/forth/Generator/forthleaky_relu_grad/GreaterEqual@gradients_1/Generator/dense/MatMul_grad/tuple/control_dependency@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros*
T0*(
_output_shapes
:         ђ
═
Cgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Select_1SelectGgradients_1/Generator/forth/Generator/forthleaky_relu_grad/GreaterEqual@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros@gradients_1/Generator/dense/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:         ђ
џ
>gradients_1/Generator/forth/Generator/forthleaky_relu_grad/SumSumAgradients_1/Generator/forth/Generator/forthleaky_relu_grad/SelectPgradients_1/Generator/forth/Generator/forthleaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
љ
Bgradients_1/Generator/forth/Generator/forthleaky_relu_grad/ReshapeReshape>gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Sum@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape*
Tshape0*(
_output_shapes
:         ђ*
T0
а
@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Sum_1SumCgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Select_1Rgradients_1/Generator/forth/Generator/forthleaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ќ
Dgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1Reshape@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Sum_1Bgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_1*(
_output_shapes
:         ђ*
T0*
Tshape0
▀
Kgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/group_depsNoOpC^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/ReshapeE^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1
ч
Sgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependencyIdentityBgradients_1/Generator/forth/Generator/forthleaky_relu_grad/ReshapeL^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/group_deps*(
_output_shapes
:         ђ*
T0*U
_classK
IGloc:@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape
Ђ
Ugradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency_1IdentityDgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1L^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ
Є
Dgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
┼
Fgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape_1Shape?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
И
Tgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ShapeFgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
њ
Bgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/MulMulSgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*(
_output_shapes
:         ђ*
T0
Б
Bgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/SumSumBgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/MulTgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
і
Fgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ReshapeReshapeBgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/SumDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ё
Dgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Mul_1Mul/Generator/forth/Generator/forthleaky_relu/alphaSgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:         ђ
Е
Dgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Sum_1SumDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Mul_1Vgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
б
Hgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape_1ReshapeDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Sum_1Fgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape_1*(
_output_shapes
:         ђ*
T0*
Tshape0
в
Ogradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/group_depsNoOpG^gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ReshapeI^gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape_1
щ
Wgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/control_dependencyIdentityFgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ReshapeP^gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape*
_output_shapes
: 
Љ
Ygradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/control_dependency_1IdentityHgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape_1P^gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:         ђ*
T0*[
_classQ
OMloc:@gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape_1
р
gradients_1/AddN_2AddNUgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency_1Ygradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:         ђ*
T0*W
_classM
KIloc:@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1
Н
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/ShapeShape?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1*
_output_shapes
:*
T0*
out_type0
Б
Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Shape_1Const*
valueB:ђ*
dtype0*
_output_shapes
:
Ь
fgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/ShapeXgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ќ
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/SumSumgradients_1/AddN_2fgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
м
Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/ReshapeReshapeTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/SumVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         ђ
Џ
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_2hgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╦
Zgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1ReshapeVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Sum_1Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:ђ
А
agradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/group_depsNoOpY^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape[^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1
М
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependencyIdentityXgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshapeb^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape*(
_output_shapes
:         ђ
╠
kgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1IdentityZgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1b^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:ђ
╠
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/ShapeShape6Generator/forth/Generator/forthfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
Б
Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Shape_1Const*
_output_shapes
:*
valueB:ђ*
dtype0
Ь
fgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/ShapeXgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
И
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/MulMuligradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency=Generator/forth/Generator/forthbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:         ђ
┘
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/SumSumTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Mulfgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
м
Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/ReshapeReshapeTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/SumVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Shape*(
_output_shapes
:         ђ*
T0*
Tshape0
│
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Mul_1Mul6Generator/forth/Generator/forthfully_connected/BiasAddigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency*(
_output_shapes
:         ђ*
T0
▀
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Sum_1SumVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Mul_1hgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
╦
Zgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1ReshapeVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Sum_1Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Shape_1*
_output_shapes	
:ђ*
T0*
Tshape0
А
agradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/group_depsNoOpY^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape[^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1
М
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyIdentityXgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshapeb^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:         ђ
╠
kgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityZgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1b^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*m
_classc
a_loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:ђ*
T0
В
Rgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/NegNegkgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:ђ
ф
_gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/group_depsNoOpl^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1S^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/Neg
О
ggradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependencyIdentitykgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1`^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:ђ
И
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1IdentityRgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/Neg`^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/Neg*
_output_shapes	
:ђ
і
Sgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:ђ
б
Xgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/group_depsNoOpj^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyT^gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/BiasAddGrad
м
`gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependencyIdentityigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyY^gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:         ђ*
T0*k
_classa
_]loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape
г
bgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/BiasAddGradY^gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:ђ
Ф
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/MulMuligradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1=Generator/forth/Generator/forthbatch_normalized/batchnorm/mul*
_output_shapes	
:ђ*
T0
░
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul_1Muligradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1@Generator/forth/Generator/forthbatch_normalized/moving_mean/read*
_output_shapes	
:ђ*
T0
Ў
agradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/group_depsNoOpU^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/MulW^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul_1
Й
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependencyIdentityTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mulb^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*g
_class]
[Yloc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul*
_output_shapes	
:ђ*
T0
─
kgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul_1b^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:ђ*
T0*i
_class_
][loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul_1
╬
Mgradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMulMatMul`gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependency:Generator/forth/Generator/forthfully_connected/kernel/read*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b(*
T0
и
Ogradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul_1MatMul)Generator/third/Generator/thirdleaky_relu`gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
ђђ*
transpose_a(*
transpose_b( *
T0
Ђ
Wgradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/group_depsNoOpN^gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMulP^gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul_1
Е
_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependencyIdentityMgradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMulX^gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/group_deps*`
_classV
TRloc:@gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul*(
_output_shapes
:         ђ*
T0
Д
agradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul_1X^gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
ђђ
њ
gradients_1/AddN_3AddNkgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1kgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1*
_output_shapes	
:ђ*
T0*m
_classc
a_loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
N
¤
Rgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/MulMulgradients_1/AddN_3:Generator/forth/Generator/forthbatch_normalized/gamma/read*
T0*
_output_shapes	
:ђ
о
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_3?Generator/forth/Generator/forthbatch_normalized/batchnorm/Rsqrt*
_output_shapes	
:ђ*
T0
Њ
_gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/group_depsNoOpS^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/MulU^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1
Х
ggradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/control_dependencyIdentityRgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul`^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul*
_output_shapes	
:ђ
╝
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1`^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1*
_output_shapes	
:ђ
Г
@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/ShapeShape-Generator/third/Generator/thirdleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
┴
Bgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_1Shape?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
р
Bgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_2Shape_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
І
Fgradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ў
@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/zerosFillBgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_2Fgradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:         ђ
Щ
Ggradients_1/Generator/third/Generator/thirdleaky_relu_grad/GreaterEqualGreaterEqual-Generator/third/Generator/thirdleaky_relu/mul?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:         ђ
г
Pgradients_1/Generator/third/Generator/thirdleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/ShapeBgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ж
Agradients_1/Generator/third/Generator/thirdleaky_relu_grad/SelectSelectGgradients_1/Generator/third/Generator/thirdleaky_relu_grad/GreaterEqual_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros*
T0*(
_output_shapes
:         ђ
В
Cgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Select_1SelectGgradients_1/Generator/third/Generator/thirdleaky_relu_grad/GreaterEqual@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:         ђ*
T0
џ
>gradients_1/Generator/third/Generator/thirdleaky_relu_grad/SumSumAgradients_1/Generator/third/Generator/thirdleaky_relu_grad/SelectPgradients_1/Generator/third/Generator/thirdleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
љ
Bgradients_1/Generator/third/Generator/thirdleaky_relu_grad/ReshapeReshape>gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Sum@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         ђ
а
@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Sum_1SumCgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Select_1Rgradients_1/Generator/third/Generator/thirdleaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ќ
Dgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1Reshape@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Sum_1Bgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:         ђ
▀
Kgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/group_depsNoOpC^gradients_1/Generator/third/Generator/thirdleaky_relu_grad/ReshapeE^gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1
ч
Sgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependencyIdentityBgradients_1/Generator/third/Generator/thirdleaky_relu_grad/ReshapeL^gradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape*(
_output_shapes
:         ђ
Ђ
Ugradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency_1IdentityDgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1L^gradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ
Є
Dgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
┼
Fgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Shape_1Shape?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
И
Tgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ShapeFgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
њ
Bgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/MulMulSgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*(
_output_shapes
:         ђ*
T0
Б
Bgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/SumSumBgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/MulTgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
і
Fgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ReshapeReshapeBgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/SumDgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
ё
Dgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Mul_1Mul/Generator/third/Generator/thirdleaky_relu/alphaSgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency*(
_output_shapes
:         ђ*
T0
Е
Dgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Sum_1SumDgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Mul_1Vgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
б
Hgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1ReshapeDgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Sum_1Fgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Shape_1*(
_output_shapes
:         ђ*
T0*
Tshape0
в
Ogradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/group_depsNoOpG^gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ReshapeI^gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1
щ
Wgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/control_dependencyIdentityFgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ReshapeP^gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape*
_output_shapes
: 
Љ
Ygradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/control_dependency_1IdentityHgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1P^gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:         ђ*
T0*[
_classQ
OMloc:@gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1
р
gradients_1/AddN_4AddNUgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency_1Ygradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/control_dependency_1*
T0*W
_classM
KIloc:@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:         ђ
Н
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/ShapeShape?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
Б
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape_1Const*
_output_shapes
:*
valueB:ђ*
dtype0
Ь
fgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/ShapeXgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ќ
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/SumSumgradients_1/AddN_4fgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
м
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/ReshapeReshapeTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/SumVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         ђ
Џ
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_4hgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
╦
Zgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1ReshapeVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Sum_1Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape_1*
_output_shapes	
:ђ*
T0*
Tshape0
А
agradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/group_depsNoOpY^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape[^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1
М
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependencyIdentityXgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshapeb^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*(
_output_shapes
:         ђ*
T0*k
_classa
_]loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape
╠
kgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1IdentityZgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1b^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:ђ
╠
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/ShapeShape6Generator/third/Generator/thirdfully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
Б
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Shape_1Const*
valueB:ђ*
dtype0*
_output_shapes
:
Ь
fgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/ShapeXgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
И
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/MulMuligradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency=Generator/third/Generator/thirdbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:         ђ
┘
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/SumSumTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Mulfgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
м
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/ReshapeReshapeTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/SumVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         ђ
│
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Mul_1Mul6Generator/third/Generator/thirdfully_connected/BiasAddigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency*(
_output_shapes
:         ђ*
T0
▀
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Sum_1SumVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Mul_1hgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
╦
Zgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1ReshapeVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Sum_1Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Shape_1*
Tshape0*
_output_shapes	
:ђ*
T0
А
agradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/group_depsNoOpY^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape[^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1
М
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyIdentityXgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshapeb^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*(
_output_shapes
:         ђ*
T0*k
_classa
_]loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape
╠
kgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityZgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1b^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*
_output_shapes	
:ђ*
T0*m
_classc
a_loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1
В
Rgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/NegNegkgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1*
_output_shapes	
:ђ*
T0
ф
_gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/group_depsNoOpl^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1S^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/Neg
О
ggradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependencyIdentitykgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1`^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:ђ
И
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1IdentityRgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/Neg`^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
_output_shapes	
:ђ*
T0*e
_class[
YWloc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/Neg
і
Sgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:ђ
б
Xgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/group_depsNoOpj^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyT^gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/BiasAddGrad
м
`gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependencyIdentityigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyY^gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/group_deps*k
_classa
_]loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:         ђ*
T0
г
bgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/BiasAddGradY^gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:ђ*
T0*f
_class\
ZXloc:@gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/BiasAddGrad
Ф
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/MulMuligradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1=Generator/third/Generator/thirdbatch_normalized/batchnorm/mul*
T0*
_output_shapes	
:ђ
░
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul_1Muligradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1@Generator/third/Generator/thirdbatch_normalized/moving_mean/read*
_output_shapes	
:ђ*
T0
Ў
agradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/group_depsNoOpU^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/MulW^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul_1
Й
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependencyIdentityTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mulb^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:ђ*
T0*g
_class]
[Yloc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul
─
kgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul_1b^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:ђ
╬
Mgradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMulMatMul`gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency:Generator/third/Generator/thirdfully_connected/kernel/read*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b(*
T0
╣
Ogradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul_1MatMul+Generator/second/Generator/secondleaky_relu`gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
ђђ*
transpose_a(*
transpose_b( 
Ђ
Wgradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/group_depsNoOpN^gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMulP^gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul_1
Е
_gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependencyIdentityMgradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMulX^gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul*(
_output_shapes
:         ђ
Д
agradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul_1X^gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
ђђ
њ
gradients_1/AddN_5AddNkgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1kgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*m
_classc
a_loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:ђ
¤
Rgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/MulMulgradients_1/AddN_5:Generator/third/Generator/thirdbatch_normalized/gamma/read*
T0*
_output_shapes	
:ђ
о
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_5?Generator/third/Generator/thirdbatch_normalized/batchnorm/Rsqrt*
T0*
_output_shapes	
:ђ
Њ
_gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/group_depsNoOpS^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/MulU^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul_1
Х
ggradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/control_dependencyIdentityRgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul`^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul*
_output_shapes	
:ђ
╝
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul_1`^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul_1*
_output_shapes	
:ђ
▒
Bgradients_1/Generator/second/Generator/secondleaky_relu_grad/ShapeShape/Generator/second/Generator/secondleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
┼
Dgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_1ShapeAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
с
Dgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_2Shape_gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependency*
out_type0*
_output_shapes
:*
T0
Ї
Hgradients_1/Generator/second/Generator/secondleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ъ
Bgradients_1/Generator/second/Generator/secondleaky_relu_grad/zerosFillDgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_2Hgradients_1/Generator/second/Generator/secondleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:         ђ
ђ
Igradients_1/Generator/second/Generator/secondleaky_relu_grad/GreaterEqualGreaterEqual/Generator/second/Generator/secondleaky_relu/mulAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*(
_output_shapes
:         ђ*
T0
▓
Rgradients_1/Generator/second/Generator/secondleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsBgradients_1/Generator/second/Generator/secondleaky_relu_grad/ShapeDgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:         :         
­
Cgradients_1/Generator/second/Generator/secondleaky_relu_grad/SelectSelectIgradients_1/Generator/second/Generator/secondleaky_relu_grad/GreaterEqual_gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependencyBgradients_1/Generator/second/Generator/secondleaky_relu_grad/zeros*
T0*(
_output_shapes
:         ђ
Ы
Egradients_1/Generator/second/Generator/secondleaky_relu_grad/Select_1SelectIgradients_1/Generator/second/Generator/secondleaky_relu_grad/GreaterEqualBgradients_1/Generator/second/Generator/secondleaky_relu_grad/zeros_gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:         ђ
а
@gradients_1/Generator/second/Generator/secondleaky_relu_grad/SumSumCgradients_1/Generator/second/Generator/secondleaky_relu_grad/SelectRgradients_1/Generator/second/Generator/secondleaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ќ
Dgradients_1/Generator/second/Generator/secondleaky_relu_grad/ReshapeReshape@gradients_1/Generator/second/Generator/secondleaky_relu_grad/SumBgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         ђ
д
Bgradients_1/Generator/second/Generator/secondleaky_relu_grad/Sum_1SumEgradients_1/Generator/second/Generator/secondleaky_relu_grad/Select_1Tgradients_1/Generator/second/Generator/secondleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ю
Fgradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1ReshapeBgradients_1/Generator/second/Generator/secondleaky_relu_grad/Sum_1Dgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:         ђ
т
Mgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/group_depsNoOpE^gradients_1/Generator/second/Generator/secondleaky_relu_grad/ReshapeG^gradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1
Ѓ
Ugradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependencyIdentityDgradients_1/Generator/second/Generator/secondleaky_relu_grad/ReshapeN^gradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/group_deps*(
_output_shapes
:         ђ*
T0*W
_classM
KIloc:@gradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape
Ѕ
Wgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependency_1IdentityFgradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1N^gradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ
Ѕ
Fgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
╔
Hgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape_1ShapeAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
out_type0*
_output_shapes
:*
T0
Й
Vgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ShapeHgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
ў
Dgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/MulMulUgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependencyAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:         ђ
Е
Dgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/SumSumDgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/MulVgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
љ
Hgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ReshapeReshapeDgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/SumFgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape*
Tshape0*
_output_shapes
: *
T0
і
Fgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Mul_1Mul1Generator/second/Generator/secondleaky_relu/alphaUgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:         ђ
»
Fgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Sum_1SumFgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Mul_1Xgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
е
Jgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape_1ReshapeFgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Sum_1Hgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape_1*(
_output_shapes
:         ђ*
T0*
Tshape0
ы
Qgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/group_depsNoOpI^gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ReshapeK^gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape_1
Ђ
Ygradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/control_dependencyIdentityHgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ReshapeR^gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape*
_output_shapes
: 
Ў
[gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/control_dependency_1IdentityJgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape_1R^gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:         ђ*
T0*]
_classS
QOloc:@gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape_1
у
gradients_1/AddN_6AddNWgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependency_1[gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/control_dependency_1*
T0*Y
_classO
MKloc:@gradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:         ђ
┘
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/ShapeShapeAGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
Ц
Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape_1Const*
_output_shapes
:*
valueB:ђ*
dtype0
З
hgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/ShapeZgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Џ
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/SumSumgradients_1/AddN_6hgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
п
Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/ReshapeReshapeVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/SumXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         ђ
Ъ
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_6jgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Л
\gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1ReshapeXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Sum_1Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape_1*
_output_shapes	
:ђ*
T0*
Tshape0
Д
cgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/group_depsNoOp[^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape]^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1
█
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependencyIdentityZgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshaped^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape*(
_output_shapes
:         ђ
н
mgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1Identity\gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1d^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*
T0*o
_classe
caloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:ђ
л
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/ShapeShape8Generator/second/Generator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ц
Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Shape_1Const*
valueB:ђ*
dtype0*
_output_shapes
:
З
hgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/ShapeZgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Й
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/MulMulkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency?Generator/second/Generator/secondbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:         ђ
▀
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/SumSumVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Mulhgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
п
Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/ReshapeReshapeVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/SumXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Shape*
Tshape0*(
_output_shapes
:         ђ*
T0
╣
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Mul_1Mul8Generator/second/Generator/secondfully_connected/BiasAddkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:         ђ
т
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Sum_1SumXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Mul_1jgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Л
\gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1ReshapeXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Sum_1Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:ђ
Д
cgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/group_depsNoOp[^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape]^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1
█
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyIdentityZgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshaped^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:         ђ
н
mgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1Identity\gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1d^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*o
_classe
caloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:ђ*
T0
­
Tgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/NegNegmgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1*
_output_shapes	
:ђ*
T0
░
agradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/group_depsNoOpn^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1U^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/Neg
▀
igradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependencyIdentitymgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1b^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
T0*o
_classe
caloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:ђ
└
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/Negb^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/Neg*
_output_shapes	
:ђ
ј
Ugradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency*
_output_shapes	
:ђ*
T0*
data_formatNHWC
е
Zgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/group_depsNoOpl^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyV^gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/BiasAddGrad
┌
bgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitykgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency[^gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:         ђ
┤
dgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityUgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/BiasAddGrad[^gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:ђ
▒
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/MulMulkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1?Generator/second/Generator/secondbatch_normalized/batchnorm/mul*
_output_shapes	
:ђ*
T0
Х
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul_1Mulkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1BGenerator/second/Generator/secondbatch_normalized/moving_mean/read*
_output_shapes	
:ђ*
T0
Ъ
cgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/group_depsNoOpW^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/MulY^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul_1
к
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependencyIdentityVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Muld^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul*
_output_shapes	
:ђ
╠
mgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul_1d^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:ђ
н
Ogradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMulMatMulbgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependency<Generator/second/Generator/secondfully_connected/kernel/read*(
_output_shapes
:         ђ*
transpose_a( *
transpose_b(*
T0
╗
Qgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul_1MatMul)Generator/first/Generator/firstleaky_relubgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
ђђ*
transpose_a(*
transpose_b( *
T0
Є
Ygradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/group_depsNoOpP^gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMulR^gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul_1
▒
agradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependencyIdentityOgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMulZ^gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul*(
_output_shapes
:         ђ
»
cgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency_1IdentityQgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul_1Z^gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
ђђ
ў
gradients_1/AddN_7AddNmgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1mgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*o
_classe
caloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:ђ
М
Tgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/MulMulgradients_1/AddN_7<Generator/second/Generator/secondbatch_normalized/gamma/read*
T0*
_output_shapes	
:ђ
┌
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_7AGenerator/second/Generator/secondbatch_normalized/batchnorm/Rsqrt*
_output_shapes	
:ђ*
T0
Ў
agradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/group_depsNoOpU^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/MulW^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul_1
Й
igradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/control_dependencyIdentityTgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mulb^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
_output_shapes	
:ђ*
T0*g
_class]
[Yloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul
─
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1IdentityVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul_1b^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul_1*
_output_shapes	
:ђ
Г
@gradients_1/Generator/first/Generator/firstleaky_relu_grad/ShapeShape-Generator/first/Generator/firstleaky_relu/mul*
_output_shapes
:*
T0*
out_type0
И
Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_1Shape6Generator/first/Generator/firstfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
с
Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_2Shapeagradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency*
out_type0*
_output_shapes
:*
T0
І
Fgradients_1/Generator/first/Generator/firstleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ў
@gradients_1/Generator/first/Generator/firstleaky_relu_grad/zerosFillBgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_2Fgradients_1/Generator/first/Generator/firstleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:         ђ
ы
Ggradients_1/Generator/first/Generator/firstleaky_relu_grad/GreaterEqualGreaterEqual-Generator/first/Generator/firstleaky_relu/mul6Generator/first/Generator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
г
Pgradients_1/Generator/first/Generator/firstleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients_1/Generator/first/Generator/firstleaky_relu_grad/ShapeBgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:         :         
В
Agradients_1/Generator/first/Generator/firstleaky_relu_grad/SelectSelectGgradients_1/Generator/first/Generator/firstleaky_relu_grad/GreaterEqualagradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency@gradients_1/Generator/first/Generator/firstleaky_relu_grad/zeros*(
_output_shapes
:         ђ*
T0
Ь
Cgradients_1/Generator/first/Generator/firstleaky_relu_grad/Select_1SelectGgradients_1/Generator/first/Generator/firstleaky_relu_grad/GreaterEqual@gradients_1/Generator/first/Generator/firstleaky_relu_grad/zerosagradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:         ђ
џ
>gradients_1/Generator/first/Generator/firstleaky_relu_grad/SumSumAgradients_1/Generator/first/Generator/firstleaky_relu_grad/SelectPgradients_1/Generator/first/Generator/firstleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
љ
Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/ReshapeReshape>gradients_1/Generator/first/Generator/firstleaky_relu_grad/Sum@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:         ђ
а
@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Sum_1SumCgradients_1/Generator/first/Generator/firstleaky_relu_grad/Select_1Rgradients_1/Generator/first/Generator/firstleaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ќ
Dgradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1Reshape@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Sum_1Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:         ђ
▀
Kgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/group_depsNoOpC^gradients_1/Generator/first/Generator/firstleaky_relu_grad/ReshapeE^gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1
ч
Sgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependencyIdentityBgradients_1/Generator/first/Generator/firstleaky_relu_grad/ReshapeL^gradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/group_deps*U
_classK
IGloc:@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape*(
_output_shapes
:         ђ*
T0
Ђ
Ugradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependency_1IdentityDgradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1L^gradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ
Є
Dgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
╝
Fgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape_1Shape6Generator/first/Generator/firstfully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
И
Tgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ShapeFgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ѕ
Bgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/MulMulSgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependency6Generator/first/Generator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:         ђ
Б
Bgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/SumSumBgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/MulTgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
і
Fgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ReshapeReshapeBgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/SumDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape*
Tshape0*
_output_shapes
: *
T0
ё
Dgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Mul_1Mul/Generator/first/Generator/firstleaky_relu/alphaSgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:         ђ
Е
Dgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Sum_1SumDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Mul_1Vgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
б
Hgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape_1ReshapeDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Sum_1Fgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:         ђ
в
Ogradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/group_depsNoOpG^gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ReshapeI^gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape_1
щ
Wgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/control_dependencyIdentityFgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ReshapeP^gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/group_deps*Y
_classO
MKloc:@gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape*
_output_shapes
: *
T0
Љ
Ygradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityHgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape_1P^gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/group_deps*[
_classQ
OMloc:@gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:         ђ*
T0
р
gradients_1/AddN_8AddNUgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependency_1Ygradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/control_dependency_1*
T0*W
_classM
KIloc:@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:         ђ
│
Sgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_8*
T0*
data_formatNHWC*
_output_shapes	
:ђ
╦
Xgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_8T^gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/BiasAddGrad
у
`gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_8Y^gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:         ђ
г
bgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/BiasAddGradY^gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:ђ
═
Mgradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMulMatMul`gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependency:Generator/first/Generator/firstfully_connected/kernel/read*
T0*'
_output_shapes
:         d*
transpose_a( *
transpose_b(
ю
Ogradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1MatMulGenerator/noise`gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	dђ*
transpose_a(*
transpose_b( 
Ђ
Wgradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/group_depsNoOpN^gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMulP^gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1
е
_gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentityMgradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMulX^gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/group_deps*`
_classV
TRloc:@gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul*'
_output_shapes
:         d*
T0
д
agradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1X^gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1*
_output_shapes
:	dђ
Ѕ
beta1_power_1/initial_valueConst*'
_class
loc:@Generator/dense/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
џ
beta1_power_1
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *'
_class
loc:@Generator/dense/bias
й
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*'
_class
loc:@Generator/dense/bias
w
beta1_power_1/readIdentitybeta1_power_1*
T0*'
_class
loc:@Generator/dense/bias*
_output_shapes
: 
Ѕ
beta2_power_1/initial_valueConst*'
_class
loc:@Generator/dense/bias*
valueB
 *wЙ?*
dtype0*
_output_shapes
: 
џ
beta2_power_1
VariableV2*
shared_name *'
_class
loc:@Generator/dense/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
й
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
use_locking(*
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes
: 
w
beta2_power_1/readIdentitybeta2_power_1*
_output_shapes
: *
T0*'
_class
loc:@Generator/dense/bias
э
\Generator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB"d   ђ   *
dtype0*
_output_shapes
:
р
RGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ї
LGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zerosFill\Generator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorRGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros/Const*
_output_shapes
:	dђ*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*

index_type0
Щ
:Generator/first/Generator/firstfully_connected/kernel/Adam
VariableV2*
shape:	dђ*
dtype0*
_output_shapes
:	dђ*
shared_name *H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
	container 
Ы
AGenerator/first/Generator/firstfully_connected/kernel/Adam/AssignAssign:Generator/first/Generator/firstfully_connected/kernel/AdamLGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros*
_output_shapes
:	dђ*
use_locking(*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
validate_shape(
ч
?Generator/first/Generator/firstfully_connected/kernel/Adam/readIdentity:Generator/first/Generator/firstfully_connected/kernel/Adam*
_output_shapes
:	dђ*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel
щ
^Generator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB"d   ђ   *
dtype0*
_output_shapes
:
с
TGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB
 *    *
dtype0
њ
NGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zerosFill^Generator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorTGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes
:	dђ*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*

index_type0
Ч
<Generator/first/Generator/firstfully_connected/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes
:	dђ*
shared_name *H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
	container *
shape:	dђ
Э
CGenerator/first/Generator/firstfully_connected/kernel/Adam_1/AssignAssign<Generator/first/Generator/firstfully_connected/kernel/Adam_1NGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
validate_shape(*
_output_shapes
:	dђ*
use_locking(*
T0
 
AGenerator/first/Generator/firstfully_connected/kernel/Adam_1/readIdentity<Generator/first/Generator/firstfully_connected/kernel/Adam_1*
_output_shapes
:	dђ*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel
р
JGenerator/first/Generator/firstfully_connected/bias/Adam/Initializer/zerosConst*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
Ь
8Generator/first/Generator/firstfully_connected/bias/Adam
VariableV2*
shared_name *F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ
Т
?Generator/first/Generator/firstfully_connected/bias/Adam/AssignAssign8Generator/first/Generator/firstfully_connected/bias/AdamJGenerator/first/Generator/firstfully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:ђ
ы
=Generator/first/Generator/firstfully_connected/bias/Adam/readIdentity8Generator/first/Generator/firstfully_connected/bias/Adam*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
_output_shapes	
:ђ
с
LGenerator/first/Generator/firstfully_connected/bias/Adam_1/Initializer/zerosConst*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
­
:Generator/first/Generator/firstfully_connected/bias/Adam_1
VariableV2*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name 
В
AGenerator/first/Generator/firstfully_connected/bias/Adam_1/AssignAssign:Generator/first/Generator/firstfully_connected/bias/Adam_1LGenerator/first/Generator/firstfully_connected/bias/Adam_1/Initializer/zeros*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
ш
?Generator/first/Generator/firstfully_connected/bias/Adam_1/readIdentity:Generator/first/Generator/firstfully_connected/bias/Adam_1*
_output_shapes	
:ђ*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias
ч
^Generator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB"ђ      *
dtype0*
_output_shapes
:
т
TGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros/ConstConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ћ
NGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zerosFill^Generator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorTGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros/Const*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*

index_type0* 
_output_shapes
:
ђђ
ђ
<Generator/second/Generator/secondfully_connected/kernel/Adam
VariableV2*
shape:
ђђ*
dtype0* 
_output_shapes
:
ђђ*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
	container 
ч
CGenerator/second/Generator/secondfully_connected/kernel/Adam/AssignAssign<Generator/second/Generator/secondfully_connected/kernel/AdamNGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(
ѓ
AGenerator/second/Generator/secondfully_connected/kernel/Adam/readIdentity<Generator/second/Generator/secondfully_connected/kernel/Adam*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:
ђђ
§
`Generator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB"ђ      *
dtype0
у
VGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Џ
PGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zerosFill`Generator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorVGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*

index_type0* 
_output_shapes
:
ђђ
ѓ
>Generator/second/Generator/secondfully_connected/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
ђђ*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
	container *
shape:
ђђ
Ђ
EGenerator/second/Generator/secondfully_connected/kernel/Adam_1/AssignAssign>Generator/second/Generator/secondfully_connected/kernel/Adam_1PGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:
ђђ
є
CGenerator/second/Generator/secondfully_connected/kernel/Adam_1/readIdentity>Generator/second/Generator/secondfully_connected/kernel/Adam_1*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:
ђђ
т
LGenerator/second/Generator/secondfully_connected/bias/Adam/Initializer/zerosConst*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
Ы
:Generator/second/Generator/secondfully_connected/bias/Adam
VariableV2*
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
	container 
Ь
AGenerator/second/Generator/secondfully_connected/bias/Adam/AssignAssign:Generator/second/Generator/secondfully_connected/bias/AdamLGenerator/second/Generator/secondfully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:ђ
э
?Generator/second/Generator/secondfully_connected/bias/Adam/readIdentity:Generator/second/Generator/secondfully_connected/bias/Adam*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
_output_shapes	
:ђ
у
NGenerator/second/Generator/secondfully_connected/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:ђ*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
valueBђ*    
З
<Generator/second/Generator/secondfully_connected/bias/Adam_1
VariableV2*
shared_name *H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ
З
CGenerator/second/Generator/secondfully_connected/bias/Adam_1/AssignAssign<Generator/second/Generator/secondfully_connected/bias/Adam_1NGenerator/second/Generator/secondfully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:ђ
ч
AGenerator/second/Generator/secondfully_connected/bias/Adam_1/readIdentity<Generator/second/Generator/secondfully_connected/bias/Adam_1*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
_output_shapes	
:ђ
ж
NGenerator/second/Generator/secondbatch_normalized/gamma/Adam/Initializer/zerosConst*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
Ш
<Generator/second/Generator/secondbatch_normalized/gamma/Adam
VariableV2*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma
Ш
CGenerator/second/Generator/secondbatch_normalized/gamma/Adam/AssignAssign<Generator/second/Generator/secondbatch_normalized/gamma/AdamNGenerator/second/Generator/secondbatch_normalized/gamma/Adam/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:ђ
§
AGenerator/second/Generator/secondbatch_normalized/gamma/Adam/readIdentity<Generator/second/Generator/secondbatch_normalized/gamma/Adam*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
_output_shapes	
:ђ
в
PGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/Initializer/zerosConst*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
Э
>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1
VariableV2*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ
Ч
EGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/AssignAssign>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1PGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/Initializer/zeros*
_output_shapes	
:ђ*
use_locking(*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
validate_shape(
Ђ
CGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/readIdentity>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
_output_shapes	
:ђ
у
MGenerator/second/Generator/secondbatch_normalized/beta/Adam/Initializer/zerosConst*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
З
;Generator/second/Generator/secondbatch_normalized/beta/Adam
VariableV2*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta
Ы
BGenerator/second/Generator/secondbatch_normalized/beta/Adam/AssignAssign;Generator/second/Generator/secondbatch_normalized/beta/AdamMGenerator/second/Generator/secondbatch_normalized/beta/Adam/Initializer/zeros*
use_locking(*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
validate_shape(*
_output_shapes	
:ђ
Щ
@Generator/second/Generator/secondbatch_normalized/beta/Adam/readIdentity;Generator/second/Generator/secondbatch_normalized/beta/Adam*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
_output_shapes	
:ђ
ж
OGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/Initializer/zerosConst*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
Ш
=Generator/second/Generator/secondbatch_normalized/beta/Adam_1
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
	container *
shape:ђ
Э
DGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/AssignAssign=Generator/second/Generator/secondbatch_normalized/beta/Adam_1OGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
validate_shape(*
_output_shapes	
:ђ
■
BGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/readIdentity=Generator/second/Generator/secondbatch_normalized/beta/Adam_1*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
_output_shapes	
:ђ
э
\Generator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB"      *
dtype0
р
RGenerator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ї
LGenerator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zerosFill\Generator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorRGenerator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros/Const*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*

index_type0* 
_output_shapes
:
ђђ*
T0
Ч
:Generator/third/Generator/thirdfully_connected/kernel/Adam
VariableV2*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
	container *
shape:
ђђ*
dtype0* 
_output_shapes
:
ђђ
з
AGenerator/third/Generator/thirdfully_connected/kernel/Adam/AssignAssign:Generator/third/Generator/thirdfully_connected/kernel/AdamLGenerator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(
Ч
?Generator/third/Generator/thirdfully_connected/kernel/Adam/readIdentity:Generator/third/Generator/thirdfully_connected/kernel/Adam*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel* 
_output_shapes
:
ђђ*
T0
щ
^Generator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
с
TGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Њ
NGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zerosFill^Generator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorTGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*

index_type0* 
_output_shapes
:
ђђ
■
<Generator/third/Generator/thirdfully_connected/kernel/Adam_1
VariableV2*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
	container *
shape:
ђђ*
dtype0* 
_output_shapes
:
ђђ*
shared_name 
щ
CGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/AssignAssign<Generator/third/Generator/thirdfully_connected/kernel/Adam_1NGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(*
T0
ђ
AGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/readIdentity<Generator/third/Generator/thirdfully_connected/kernel/Adam_1*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel* 
_output_shapes
:
ђђ
р
JGenerator/third/Generator/thirdfully_connected/bias/Adam/Initializer/zerosConst*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
Ь
8Generator/third/Generator/thirdfully_connected/bias/Adam
VariableV2*
shared_name *F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ
Т
?Generator/third/Generator/thirdfully_connected/bias/Adam/AssignAssign8Generator/third/Generator/thirdfully_connected/bias/AdamJGenerator/third/Generator/thirdfully_connected/bias/Adam/Initializer/zeros*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0
ы
=Generator/third/Generator/thirdfully_connected/bias/Adam/readIdentity8Generator/third/Generator/thirdfully_connected/bias/Adam*
_output_shapes	
:ђ*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias
с
LGenerator/third/Generator/thirdfully_connected/bias/Adam_1/Initializer/zerosConst*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
­
:Generator/third/Generator/thirdfully_connected/bias/Adam_1
VariableV2*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name 
В
AGenerator/third/Generator/thirdfully_connected/bias/Adam_1/AssignAssign:Generator/third/Generator/thirdfully_connected/bias/Adam_1LGenerator/third/Generator/thirdfully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
validate_shape(*
_output_shapes	
:ђ
ш
?Generator/third/Generator/thirdfully_connected/bias/Adam_1/readIdentity:Generator/third/Generator/thirdfully_connected/bias/Adam_1*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
_output_shapes	
:ђ
т
LGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:ђ*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
valueBђ*    
Ы
:Generator/third/Generator/thirdbatch_normalized/gamma/Adam
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
	container *
shape:ђ
Ь
AGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/AssignAssign:Generator/third/Generator/thirdbatch_normalized/gamma/AdamLGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/Initializer/zeros*
_output_shapes	
:ђ*
use_locking(*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
validate_shape(
э
?Generator/third/Generator/thirdbatch_normalized/gamma/Adam/readIdentity:Generator/third/Generator/thirdbatch_normalized/gamma/Adam*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
_output_shapes	
:ђ
у
NGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/Initializer/zerosConst*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
З
<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
	container *
shape:ђ
З
CGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/AssignAssign<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1NGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:ђ
ч
AGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/readIdentity<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
_output_shapes	
:ђ
с
KGenerator/third/Generator/thirdbatch_normalized/beta/Adam/Initializer/zerosConst*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
­
9Generator/third/Generator/thirdbatch_normalized/beta/Adam
VariableV2*
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
	container 
Ж
@Generator/third/Generator/thirdbatch_normalized/beta/Adam/AssignAssign9Generator/third/Generator/thirdbatch_normalized/beta/AdamKGenerator/third/Generator/thirdbatch_normalized/beta/Adam/Initializer/zeros*
_output_shapes	
:ђ*
use_locking(*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
validate_shape(
З
>Generator/third/Generator/thirdbatch_normalized/beta/Adam/readIdentity9Generator/third/Generator/thirdbatch_normalized/beta/Adam*
_output_shapes	
:ђ*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta
т
MGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/Initializer/zerosConst*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
valueBђ*    *
dtype0*
_output_shapes	
:ђ
Ы
;Generator/third/Generator/thirdbatch_normalized/beta/Adam_1
VariableV2*
shared_name *G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ
­
BGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/AssignAssign;Generator/third/Generator/thirdbatch_normalized/beta/Adam_1MGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
validate_shape(*
_output_shapes	
:ђ
Э
@Generator/third/Generator/thirdbatch_normalized/beta/Adam_1/readIdentity;Generator/third/Generator/thirdbatch_normalized/beta/Adam_1*
_output_shapes	
:ђ*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta
э
\Generator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
р
RGenerator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ї
LGenerator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zerosFill\Generator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorRGenerator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
ђђ*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*

index_type0
Ч
:Generator/forth/Generator/forthfully_connected/kernel/Adam
VariableV2*
	container *
shape:
ђђ*
dtype0* 
_output_shapes
:
ђђ*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel
з
AGenerator/forth/Generator/forthfully_connected/kernel/Adam/AssignAssign:Generator/forth/Generator/forthfully_connected/kernel/AdamLGenerator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros* 
_output_shapes
:
ђђ*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
validate_shape(
Ч
?Generator/forth/Generator/forthfully_connected/kernel/Adam/readIdentity:Generator/forth/Generator/forthfully_connected/kernel/Adam*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:
ђђ
щ
^Generator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB"      *
dtype0
с
TGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Њ
NGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zerosFill^Generator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorTGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*

index_type0* 
_output_shapes
:
ђђ
■
<Generator/forth/Generator/forthfully_connected/kernel/Adam_1
VariableV2*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
	container *
shape:
ђђ*
dtype0* 
_output_shapes
:
ђђ
щ
CGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/AssignAssign<Generator/forth/Generator/forthfully_connected/kernel/Adam_1NGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
validate_shape(* 
_output_shapes
:
ђђ*
use_locking(*
T0
ђ
AGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/readIdentity<Generator/forth/Generator/forthfully_connected/kernel/Adam_1*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:
ђђ*
T0
ь
ZGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB:ђ
П
PGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB
 *    
ђ
JGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zerosFillZGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros/shape_as_tensorPGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros/Const*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*

index_type0*
_output_shapes	
:ђ
Ь
8Generator/forth/Generator/forthfully_connected/bias/Adam
VariableV2*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name *F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias
Т
?Generator/forth/Generator/forthfully_connected/bias/Adam/AssignAssign8Generator/forth/Generator/forthfully_connected/bias/AdamJGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
ы
=Generator/forth/Generator/forthfully_connected/bias/Adam/readIdentity8Generator/forth/Generator/forthfully_connected/bias/Adam*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
_output_shapes	
:ђ
№
\Generator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB:ђ*
dtype0
▀
RGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/ConstConst*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
є
LGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zerosFill\Generator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorRGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/Const*
_output_shapes	
:ђ*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*

index_type0
­
:Generator/forth/Generator/forthfully_connected/bias/Adam_1
VariableV2*
shared_name *F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ
В
AGenerator/forth/Generator/forthfully_connected/bias/Adam_1/AssignAssign:Generator/forth/Generator/forthfully_connected/bias/Adam_1LGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias
ш
?Generator/forth/Generator/forthfully_connected/bias/Adam_1/readIdentity:Generator/forth/Generator/forthfully_connected/bias/Adam_1*
_output_shapes	
:ђ*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias
ы
\Generator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB:ђ*
dtype0*
_output_shapes
:
р
RGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB
 *    *
dtype0*
_output_shapes
: 
ѕ
LGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zerosFill\Generator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/shape_as_tensorRGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/Const*
_output_shapes	
:ђ*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*

index_type0
Ы
:Generator/forth/Generator/forthbatch_normalized/gamma/Adam
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
	container *
shape:ђ
Ь
AGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/AssignAssign:Generator/forth/Generator/forthbatch_normalized/gamma/AdamLGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:ђ
э
?Generator/forth/Generator/forthbatch_normalized/gamma/Adam/readIdentity:Generator/forth/Generator/forthbatch_normalized/gamma/Adam*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
_output_shapes	
:ђ
з
^Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB:ђ*
dtype0*
_output_shapes
:
с
TGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB
 *    *
dtype0*
_output_shapes
: 
ј
NGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zerosFill^Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros/shape_as_tensorTGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros/Const*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*

index_type0*
_output_shapes	
:ђ
З
<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1
VariableV2*
dtype0*
_output_shapes	
:ђ*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
	container *
shape:ђ
З
CGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/AssignAssign<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1NGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:ђ*
use_locking(
ч
AGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/readIdentity<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
_output_shapes	
:ђ*
T0
№
[Generator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros/shape_as_tensorConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB:ђ*
dtype0*
_output_shapes
:
▀
QGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB
 *    *
dtype0
ё
KGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zerosFill[Generator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros/shape_as_tensorQGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros/Const*
_output_shapes	
:ђ*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*

index_type0
­
9Generator/forth/Generator/forthbatch_normalized/beta/Adam
VariableV2*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ*
shared_name 
Ж
@Generator/forth/Generator/forthbatch_normalized/beta/Adam/AssignAssign9Generator/forth/Generator/forthbatch_normalized/beta/AdamKGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:ђ*
use_locking(*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta
З
>Generator/forth/Generator/forthbatch_normalized/beta/Adam/readIdentity9Generator/forth/Generator/forthbatch_normalized/beta/Adam*
_output_shapes	
:ђ*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta
ы
]Generator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros/shape_as_tensorConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB:ђ*
dtype0*
_output_shapes
:
р
SGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros/ConstConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
і
MGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zerosFill]Generator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros/shape_as_tensorSGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros/Const*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*

index_type0*
_output_shapes	
:ђ*
T0
Ы
;Generator/forth/Generator/forthbatch_normalized/beta/Adam_1
VariableV2*
shared_name *G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
	container *
shape:ђ*
dtype0*
_output_shapes	
:ђ
­
BGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/AssignAssign;Generator/forth/Generator/forthbatch_normalized/beta/Adam_1MGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
validate_shape(*
_output_shapes	
:ђ
Э
@Generator/forth/Generator/forthbatch_normalized/beta/Adam_1/readIdentity;Generator/forth/Generator/forthbatch_normalized/beta/Adam_1*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
_output_shapes	
:ђ*
T0
╣
=Generator/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@Generator/dense/kernel*
valueB"     *
dtype0*
_output_shapes
:
Б
3Generator/dense/kernel/Adam/Initializer/zeros/ConstConst*)
_class
loc:@Generator/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Љ
-Generator/dense/kernel/Adam/Initializer/zerosFill=Generator/dense/kernel/Adam/Initializer/zeros/shape_as_tensor3Generator/dense/kernel/Adam/Initializer/zeros/Const*)
_class
loc:@Generator/dense/kernel*

index_type0* 
_output_shapes
:
ђљ*
T0
Й
Generator/dense/kernel/Adam
VariableV2*
shape:
ђљ*
dtype0* 
_output_shapes
:
ђљ*
shared_name *)
_class
loc:@Generator/dense/kernel*
	container 
э
"Generator/dense/kernel/Adam/AssignAssignGenerator/dense/kernel/Adam-Generator/dense/kernel/Adam/Initializer/zeros* 
_output_shapes
:
ђљ*
use_locking(*
T0*)
_class
loc:@Generator/dense/kernel*
validate_shape(
Ъ
 Generator/dense/kernel/Adam/readIdentityGenerator/dense/kernel/Adam*
T0*)
_class
loc:@Generator/dense/kernel* 
_output_shapes
:
ђљ
╗
?Generator/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@Generator/dense/kernel*
valueB"     *
dtype0*
_output_shapes
:
Ц
5Generator/dense/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *)
_class
loc:@Generator/dense/kernel*
valueB
 *    
Ќ
/Generator/dense/kernel/Adam_1/Initializer/zerosFill?Generator/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor5Generator/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*)
_class
loc:@Generator/dense/kernel*

index_type0* 
_output_shapes
:
ђљ
└
Generator/dense/kernel/Adam_1
VariableV2*
shape:
ђљ*
dtype0* 
_output_shapes
:
ђљ*
shared_name *)
_class
loc:@Generator/dense/kernel*
	container 
§
$Generator/dense/kernel/Adam_1/AssignAssignGenerator/dense/kernel/Adam_1/Generator/dense/kernel/Adam_1/Initializer/zeros*)
_class
loc:@Generator/dense/kernel*
validate_shape(* 
_output_shapes
:
ђљ*
use_locking(*
T0
Б
"Generator/dense/kernel/Adam_1/readIdentityGenerator/dense/kernel/Adam_1*
T0*)
_class
loc:@Generator/dense/kernel* 
_output_shapes
:
ђљ
Б
+Generator/dense/bias/Adam/Initializer/zerosConst*'
_class
loc:@Generator/dense/bias*
valueBљ*    *
dtype0*
_output_shapes	
:љ
░
Generator/dense/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:љ*
shared_name *'
_class
loc:@Generator/dense/bias*
	container *
shape:љ
Ж
 Generator/dense/bias/Adam/AssignAssignGenerator/dense/bias/Adam+Generator/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes	
:љ
ћ
Generator/dense/bias/Adam/readIdentityGenerator/dense/bias/Adam*
T0*'
_class
loc:@Generator/dense/bias*
_output_shapes	
:љ
Ц
-Generator/dense/bias/Adam_1/Initializer/zerosConst*'
_class
loc:@Generator/dense/bias*
valueBљ*    *
dtype0*
_output_shapes	
:љ
▓
Generator/dense/bias/Adam_1
VariableV2*
shape:љ*
dtype0*
_output_shapes	
:љ*
shared_name *'
_class
loc:@Generator/dense/bias*
	container 
­
"Generator/dense/bias/Adam_1/AssignAssignGenerator/dense/bias/Adam_1-Generator/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes	
:љ
ў
 Generator/dense/bias/Adam_1/readIdentityGenerator/dense/bias/Adam_1*
T0*'
_class
loc:@Generator/dense/bias*
_output_shapes	
:љ
Y
Adam_1/learning_rateConst*
valueB
 *иQ9*
dtype0*
_output_shapes
: 
Q
Adam_1/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Q
Adam_1/beta2Const*
valueB
 *wЙ?*
dtype0*
_output_shapes
: 
S
Adam_1/epsilonConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
з
MAdam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdam	ApplyAdam5Generator/first/Generator/firstfully_connected/kernel:Generator/first/Generator/firstfully_connected/kernel/Adam<Generator/first/Generator/firstfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
use_nesterov( *
_output_shapes
:	dђ
Т
KAdam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdam	ApplyAdam3Generator/first/Generator/firstfully_connected/bias8Generator/first/Generator/firstfully_connected/bias/Adam:Generator/first/Generator/firstfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
use_nesterov( *
_output_shapes	
:ђ
ђ
OAdam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdam	ApplyAdam7Generator/second/Generator/secondfully_connected/kernel<Generator/second/Generator/secondfully_connected/kernel/Adam>Generator/second/Generator/secondfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsiloncgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
use_nesterov( * 
_output_shapes
:
ђђ
Ы
MAdam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdam	ApplyAdam5Generator/second/Generator/secondfully_connected/bias:Generator/second/Generator/secondfully_connected/bias/Adam<Generator/second/Generator/secondfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilondgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
use_nesterov( *
_output_shapes	
:ђ*
use_locking( 
Ѓ
OAdam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdam	ApplyAdam7Generator/second/Generator/secondbatch_normalized/gamma<Generator/second/Generator/secondbatch_normalized/gamma/Adam>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
use_nesterov( *
_output_shapes	
:ђ
Ч
NAdam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdam	ApplyAdam6Generator/second/Generator/secondbatch_normalized/beta;Generator/second/Generator/secondbatch_normalized/beta/Adam=Generator/second/Generator/secondbatch_normalized/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonigradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependency*
use_nesterov( *
_output_shapes	
:ђ*
use_locking( *
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta
З
MAdam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam	ApplyAdam5Generator/third/Generator/thirdfully_connected/kernel:Generator/third/Generator/thirdfully_connected/kernel/Adam<Generator/third/Generator/thirdfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependency_1*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
use_nesterov( * 
_output_shapes
:
ђђ*
use_locking( 
Т
KAdam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdam	ApplyAdam3Generator/third/Generator/thirdfully_connected/bias8Generator/third/Generator/thirdfully_connected/bias/Adam:Generator/third/Generator/thirdfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:ђ*
use_locking( *
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias
э
MAdam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdam	ApplyAdam5Generator/third/Generator/thirdbatch_normalized/gamma:Generator/third/Generator/thirdbatch_normalized/gamma/Adam<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
use_nesterov( *
_output_shapes	
:ђ*
use_locking( *
T0
­
LAdam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdam	ApplyAdam4Generator/third/Generator/thirdbatch_normalized/beta9Generator/third/Generator/thirdbatch_normalized/beta/Adam;Generator/third/Generator/thirdbatch_normalized/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonggradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependency*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
use_nesterov( *
_output_shapes	
:ђ*
use_locking( *
T0
З
MAdam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdam	ApplyAdam5Generator/forth/Generator/forthfully_connected/kernel:Generator/forth/Generator/forthfully_connected/kernel/Adam<Generator/forth/Generator/forthfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
use_nesterov( * 
_output_shapes
:
ђђ
Т
KAdam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdam	ApplyAdam3Generator/forth/Generator/forthfully_connected/bias8Generator/forth/Generator/forthfully_connected/bias/Adam:Generator/forth/Generator/forthfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:ђ*
use_locking( *
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
use_nesterov( 
э
MAdam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdam	ApplyAdam5Generator/forth/Generator/forthbatch_normalized/gamma:Generator/forth/Generator/forthbatch_normalized/gamma/Adam<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
use_nesterov( *
_output_shapes	
:ђ*
use_locking( 
­
LAdam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdam	ApplyAdam4Generator/forth/Generator/forthbatch_normalized/beta9Generator/forth/Generator/forthbatch_normalized/beta/Adam;Generator/forth/Generator/forthbatch_normalized/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonggradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
use_nesterov( *
_output_shapes	
:ђ
║
.Adam_1/update_Generator/dense/kernel/ApplyAdam	ApplyAdamGenerator/dense/kernelGenerator/dense/kernel/AdamGenerator/dense/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonBgradients_1/Generator/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@Generator/dense/kernel*
use_nesterov( * 
_output_shapes
:
ђљ
г
,Adam_1/update_Generator/dense/bias/ApplyAdam	ApplyAdamGenerator/dense/biasGenerator/dense/bias/AdamGenerator/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonCgradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependency_1*'
_class
loc:@Generator/dense/bias*
use_nesterov( *
_output_shapes	
:љ*
use_locking( *
T0
║


Adam_1/mulMulbeta1_power_1/readAdam_1/beta1-^Adam_1/update_Generator/dense/bias/ApplyAdam/^Adam_1/update_Generator/dense/kernel/ApplyAdamL^Adam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdamO^Adam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdamN^Adam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam*
T0*'
_class
loc:@Generator/dense/bias*
_output_shapes
: 
Ц
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
╝

Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2-^Adam_1/update_Generator/dense/bias/ApplyAdam/^Adam_1/update_Generator/dense/kernel/ApplyAdamL^Adam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdamO^Adam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdamN^Adam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam*
_output_shapes
: *
T0*'
_class
loc:@Generator/dense/bias
Е
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
_output_shapes
: *
use_locking( *
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(
ь	
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_1-^Adam_1/update_Generator/dense/bias/ApplyAdam/^Adam_1/update_Generator/dense/kernel/ApplyAdamL^Adam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdamO^Adam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdamN^Adam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam
g
Merge/MergeSummaryMergeSummarydiscriminator_lossgenerator_loss*
N*
_output_shapes
: ""7
	summaries*
(
discriminator_loss:0
generator_loss:0"Л*
trainable_variables╣*Х*
І
7Generator/first/Generator/firstfully_connected/kernel:0<Generator/first/Generator/firstfully_connected/kernel/Assign<Generator/first/Generator/firstfully_connected/kernel/read:02RGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform:08
Щ
5Generator/first/Generator/firstfully_connected/bias:0:Generator/first/Generator/firstfully_connected/bias/Assign:Generator/first/Generator/firstfully_connected/bias/read:02GGenerator/first/Generator/firstfully_connected/bias/Initializer/zeros:08
Њ
9Generator/second/Generator/secondfully_connected/kernel:0>Generator/second/Generator/secondfully_connected/kernel/Assign>Generator/second/Generator/secondfully_connected/kernel/read:02TGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform:08
ѓ
7Generator/second/Generator/secondfully_connected/bias:0<Generator/second/Generator/secondfully_connected/bias/Assign<Generator/second/Generator/secondfully_connected/bias/read:02IGenerator/second/Generator/secondfully_connected/bias/Initializer/zeros:08
Ѕ
9Generator/second/Generator/secondbatch_normalized/gamma:0>Generator/second/Generator/secondbatch_normalized/gamma/Assign>Generator/second/Generator/secondbatch_normalized/gamma/read:02JGenerator/second/Generator/secondbatch_normalized/gamma/Initializer/ones:08
є
8Generator/second/Generator/secondbatch_normalized/beta:0=Generator/second/Generator/secondbatch_normalized/beta/Assign=Generator/second/Generator/secondbatch_normalized/beta/read:02JGenerator/second/Generator/secondbatch_normalized/beta/Initializer/zeros:08
І
7Generator/third/Generator/thirdfully_connected/kernel:0<Generator/third/Generator/thirdfully_connected/kernel/Assign<Generator/third/Generator/thirdfully_connected/kernel/read:02RGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform:08
Щ
5Generator/third/Generator/thirdfully_connected/bias:0:Generator/third/Generator/thirdfully_connected/bias/Assign:Generator/third/Generator/thirdfully_connected/bias/read:02GGenerator/third/Generator/thirdfully_connected/bias/Initializer/zeros:08
Ђ
7Generator/third/Generator/thirdbatch_normalized/gamma:0<Generator/third/Generator/thirdbatch_normalized/gamma/Assign<Generator/third/Generator/thirdbatch_normalized/gamma/read:02HGenerator/third/Generator/thirdbatch_normalized/gamma/Initializer/ones:08
■
6Generator/third/Generator/thirdbatch_normalized/beta:0;Generator/third/Generator/thirdbatch_normalized/beta/Assign;Generator/third/Generator/thirdbatch_normalized/beta/read:02HGenerator/third/Generator/thirdbatch_normalized/beta/Initializer/zeros:08
І
7Generator/forth/Generator/forthfully_connected/kernel:0<Generator/forth/Generator/forthfully_connected/kernel/Assign<Generator/forth/Generator/forthfully_connected/kernel/read:02RGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform:08
Щ
5Generator/forth/Generator/forthfully_connected/bias:0:Generator/forth/Generator/forthfully_connected/bias/Assign:Generator/forth/Generator/forthfully_connected/bias/read:02GGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros:08
Ђ
7Generator/forth/Generator/forthbatch_normalized/gamma:0<Generator/forth/Generator/forthbatch_normalized/gamma/Assign<Generator/forth/Generator/forthbatch_normalized/gamma/read:02HGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones:08
■
6Generator/forth/Generator/forthbatch_normalized/beta:0;Generator/forth/Generator/forthbatch_normalized/beta/Assign;Generator/forth/Generator/forthbatch_normalized/beta/read:02HGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros:08
Ј
Generator/dense/kernel:0Generator/dense/kernel/AssignGenerator/dense/kernel/read:023Generator/dense/kernel/Initializer/random_uniform:08
~
Generator/dense/bias:0Generator/dense/bias/AssignGenerator/dense/bias/read:02(Generator/dense/bias/Initializer/zeros:08
Ф
?Discriminator/first/Discriminator/firstfully_connected/kernel:0DDiscriminator/first/Discriminator/firstfully_connected/kernel/AssignDDiscriminator/first/Discriminator/firstfully_connected/kernel/read:02ZDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform:08
џ
=Discriminator/first/Discriminator/firstfully_connected/bias:0BDiscriminator/first/Discriminator/firstfully_connected/bias/AssignBDiscriminator/first/Discriminator/firstfully_connected/bias/read:02ODiscriminator/first/Discriminator/firstfully_connected/bias/Initializer/zeros:08
│
ADiscriminator/second/Discriminator/secondfully_connected/kernel:0FDiscriminator/second/Discriminator/secondfully_connected/kernel/AssignFDiscriminator/second/Discriminator/secondfully_connected/kernel/read:02\Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform:08
б
?Discriminator/second/Discriminator/secondfully_connected/bias:0DDiscriminator/second/Discriminator/secondfully_connected/bias/AssignDDiscriminator/second/Discriminator/secondfully_connected/bias/read:02QDiscriminator/second/Discriminator/secondfully_connected/bias/Initializer/zeros:08
Ќ
Discriminator/out/kernel:0Discriminator/out/kernel/AssignDiscriminator/out/kernel/read:025Discriminator/out/kernel/Initializer/random_uniform:08
є
Discriminator/out/bias:0Discriminator/out/bias/AssignDiscriminator/out/bias/read:02*Discriminator/out/bias/Initializer/zeros:08"
train_op

Adam
Adam_1"╝ќ
	variablesГќЕќ
І
7Generator/first/Generator/firstfully_connected/kernel:0<Generator/first/Generator/firstfully_connected/kernel/Assign<Generator/first/Generator/firstfully_connected/kernel/read:02RGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform:08
Щ
5Generator/first/Generator/firstfully_connected/bias:0:Generator/first/Generator/firstfully_connected/bias/Assign:Generator/first/Generator/firstfully_connected/bias/read:02GGenerator/first/Generator/firstfully_connected/bias/Initializer/zeros:08
Њ
9Generator/second/Generator/secondfully_connected/kernel:0>Generator/second/Generator/secondfully_connected/kernel/Assign>Generator/second/Generator/secondfully_connected/kernel/read:02TGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform:08
ѓ
7Generator/second/Generator/secondfully_connected/bias:0<Generator/second/Generator/secondfully_connected/bias/Assign<Generator/second/Generator/secondfully_connected/bias/read:02IGenerator/second/Generator/secondfully_connected/bias/Initializer/zeros:08
Ѕ
9Generator/second/Generator/secondbatch_normalized/gamma:0>Generator/second/Generator/secondbatch_normalized/gamma/Assign>Generator/second/Generator/secondbatch_normalized/gamma/read:02JGenerator/second/Generator/secondbatch_normalized/gamma/Initializer/ones:08
є
8Generator/second/Generator/secondbatch_normalized/beta:0=Generator/second/Generator/secondbatch_normalized/beta/Assign=Generator/second/Generator/secondbatch_normalized/beta/read:02JGenerator/second/Generator/secondbatch_normalized/beta/Initializer/zeros:08
а
?Generator/second/Generator/secondbatch_normalized/moving_mean:0DGenerator/second/Generator/secondbatch_normalized/moving_mean/AssignDGenerator/second/Generator/secondbatch_normalized/moving_mean/read:02QGenerator/second/Generator/secondbatch_normalized/moving_mean/Initializer/zeros:0
»
CGenerator/second/Generator/secondbatch_normalized/moving_variance:0HGenerator/second/Generator/secondbatch_normalized/moving_variance/AssignHGenerator/second/Generator/secondbatch_normalized/moving_variance/read:02TGenerator/second/Generator/secondbatch_normalized/moving_variance/Initializer/ones:0
І
7Generator/third/Generator/thirdfully_connected/kernel:0<Generator/third/Generator/thirdfully_connected/kernel/Assign<Generator/third/Generator/thirdfully_connected/kernel/read:02RGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform:08
Щ
5Generator/third/Generator/thirdfully_connected/bias:0:Generator/third/Generator/thirdfully_connected/bias/Assign:Generator/third/Generator/thirdfully_connected/bias/read:02GGenerator/third/Generator/thirdfully_connected/bias/Initializer/zeros:08
Ђ
7Generator/third/Generator/thirdbatch_normalized/gamma:0<Generator/third/Generator/thirdbatch_normalized/gamma/Assign<Generator/third/Generator/thirdbatch_normalized/gamma/read:02HGenerator/third/Generator/thirdbatch_normalized/gamma/Initializer/ones:08
■
6Generator/third/Generator/thirdbatch_normalized/beta:0;Generator/third/Generator/thirdbatch_normalized/beta/Assign;Generator/third/Generator/thirdbatch_normalized/beta/read:02HGenerator/third/Generator/thirdbatch_normalized/beta/Initializer/zeros:08
ў
=Generator/third/Generator/thirdbatch_normalized/moving_mean:0BGenerator/third/Generator/thirdbatch_normalized/moving_mean/AssignBGenerator/third/Generator/thirdbatch_normalized/moving_mean/read:02OGenerator/third/Generator/thirdbatch_normalized/moving_mean/Initializer/zeros:0
Д
AGenerator/third/Generator/thirdbatch_normalized/moving_variance:0FGenerator/third/Generator/thirdbatch_normalized/moving_variance/AssignFGenerator/third/Generator/thirdbatch_normalized/moving_variance/read:02RGenerator/third/Generator/thirdbatch_normalized/moving_variance/Initializer/ones:0
І
7Generator/forth/Generator/forthfully_connected/kernel:0<Generator/forth/Generator/forthfully_connected/kernel/Assign<Generator/forth/Generator/forthfully_connected/kernel/read:02RGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform:08
Щ
5Generator/forth/Generator/forthfully_connected/bias:0:Generator/forth/Generator/forthfully_connected/bias/Assign:Generator/forth/Generator/forthfully_connected/bias/read:02GGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros:08
Ђ
7Generator/forth/Generator/forthbatch_normalized/gamma:0<Generator/forth/Generator/forthbatch_normalized/gamma/Assign<Generator/forth/Generator/forthbatch_normalized/gamma/read:02HGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones:08
■
6Generator/forth/Generator/forthbatch_normalized/beta:0;Generator/forth/Generator/forthbatch_normalized/beta/Assign;Generator/forth/Generator/forthbatch_normalized/beta/read:02HGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros:08
ў
=Generator/forth/Generator/forthbatch_normalized/moving_mean:0BGenerator/forth/Generator/forthbatch_normalized/moving_mean/AssignBGenerator/forth/Generator/forthbatch_normalized/moving_mean/read:02OGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros:0
Д
AGenerator/forth/Generator/forthbatch_normalized/moving_variance:0FGenerator/forth/Generator/forthbatch_normalized/moving_variance/AssignFGenerator/forth/Generator/forthbatch_normalized/moving_variance/read:02RGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones:0
Ј
Generator/dense/kernel:0Generator/dense/kernel/AssignGenerator/dense/kernel/read:023Generator/dense/kernel/Initializer/random_uniform:08
~
Generator/dense/bias:0Generator/dense/bias/AssignGenerator/dense/bias/read:02(Generator/dense/bias/Initializer/zeros:08
Ф
?Discriminator/first/Discriminator/firstfully_connected/kernel:0DDiscriminator/first/Discriminator/firstfully_connected/kernel/AssignDDiscriminator/first/Discriminator/firstfully_connected/kernel/read:02ZDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform:08
џ
=Discriminator/first/Discriminator/firstfully_connected/bias:0BDiscriminator/first/Discriminator/firstfully_connected/bias/AssignBDiscriminator/first/Discriminator/firstfully_connected/bias/read:02ODiscriminator/first/Discriminator/firstfully_connected/bias/Initializer/zeros:08
│
ADiscriminator/second/Discriminator/secondfully_connected/kernel:0FDiscriminator/second/Discriminator/secondfully_connected/kernel/AssignFDiscriminator/second/Discriminator/secondfully_connected/kernel/read:02\Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform:08
б
?Discriminator/second/Discriminator/secondfully_connected/bias:0DDiscriminator/second/Discriminator/secondfully_connected/bias/AssignDDiscriminator/second/Discriminator/secondfully_connected/bias/read:02QDiscriminator/second/Discriminator/secondfully_connected/bias/Initializer/zeros:08
Ќ
Discriminator/out/kernel:0Discriminator/out/kernel/AssignDiscriminator/out/kernel/read:025Discriminator/out/kernel/Initializer/random_uniform:08
є
Discriminator/out/bias:0Discriminator/out/bias/AssignDiscriminator/out/bias/read:02*Discriminator/out/bias/Initializer/zeros:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
┤
DDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam:0IDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/AssignIDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/read:02VDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros:0
╝
FDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1:0KDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/AssignKDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/read:02XDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros:0
г
BDiscriminator/first/Discriminator/firstfully_connected/bias/Adam:0GDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/AssignGDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/read:02TDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/Initializer/zeros:0
┤
DDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1:0IDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/AssignIDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/read:02VDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/Initializer/zeros:0
╝
FDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam:0KDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/AssignKDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/read:02XDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros:0
─
HDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1:0MDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/AssignMDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/read:02ZDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros:0
┤
DDiscriminator/second/Discriminator/secondfully_connected/bias/Adam:0IDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/AssignIDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/read:02VDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/Initializer/zeros:0
╝
FDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1:0KDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/AssignKDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/read:02XDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/Initializer/zeros:0
а
Discriminator/out/kernel/Adam:0$Discriminator/out/kernel/Adam/Assign$Discriminator/out/kernel/Adam/read:021Discriminator/out/kernel/Adam/Initializer/zeros:0
е
!Discriminator/out/kernel/Adam_1:0&Discriminator/out/kernel/Adam_1/Assign&Discriminator/out/kernel/Adam_1/read:023Discriminator/out/kernel/Adam_1/Initializer/zeros:0
ў
Discriminator/out/bias/Adam:0"Discriminator/out/bias/Adam/Assign"Discriminator/out/bias/Adam/read:02/Discriminator/out/bias/Adam/Initializer/zeros:0
а
Discriminator/out/bias/Adam_1:0$Discriminator/out/bias/Adam_1/Assign$Discriminator/out/bias/Adam_1/read:021Discriminator/out/bias/Adam_1/Initializer/zeros:0
\
beta1_power_1:0beta1_power_1/Assignbeta1_power_1/read:02beta1_power_1/initial_value:0
\
beta2_power_1:0beta2_power_1/Assignbeta2_power_1/read:02beta2_power_1/initial_value:0
ћ
<Generator/first/Generator/firstfully_connected/kernel/Adam:0AGenerator/first/Generator/firstfully_connected/kernel/Adam/AssignAGenerator/first/Generator/firstfully_connected/kernel/Adam/read:02NGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros:0
ю
>Generator/first/Generator/firstfully_connected/kernel/Adam_1:0CGenerator/first/Generator/firstfully_connected/kernel/Adam_1/AssignCGenerator/first/Generator/firstfully_connected/kernel/Adam_1/read:02PGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros:0
ї
:Generator/first/Generator/firstfully_connected/bias/Adam:0?Generator/first/Generator/firstfully_connected/bias/Adam/Assign?Generator/first/Generator/firstfully_connected/bias/Adam/read:02LGenerator/first/Generator/firstfully_connected/bias/Adam/Initializer/zeros:0
ћ
<Generator/first/Generator/firstfully_connected/bias/Adam_1:0AGenerator/first/Generator/firstfully_connected/bias/Adam_1/AssignAGenerator/first/Generator/firstfully_connected/bias/Adam_1/read:02NGenerator/first/Generator/firstfully_connected/bias/Adam_1/Initializer/zeros:0
ю
>Generator/second/Generator/secondfully_connected/kernel/Adam:0CGenerator/second/Generator/secondfully_connected/kernel/Adam/AssignCGenerator/second/Generator/secondfully_connected/kernel/Adam/read:02PGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros:0
ц
@Generator/second/Generator/secondfully_connected/kernel/Adam_1:0EGenerator/second/Generator/secondfully_connected/kernel/Adam_1/AssignEGenerator/second/Generator/secondfully_connected/kernel/Adam_1/read:02RGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros:0
ћ
<Generator/second/Generator/secondfully_connected/bias/Adam:0AGenerator/second/Generator/secondfully_connected/bias/Adam/AssignAGenerator/second/Generator/secondfully_connected/bias/Adam/read:02NGenerator/second/Generator/secondfully_connected/bias/Adam/Initializer/zeros:0
ю
>Generator/second/Generator/secondfully_connected/bias/Adam_1:0CGenerator/second/Generator/secondfully_connected/bias/Adam_1/AssignCGenerator/second/Generator/secondfully_connected/bias/Adam_1/read:02PGenerator/second/Generator/secondfully_connected/bias/Adam_1/Initializer/zeros:0
ю
>Generator/second/Generator/secondbatch_normalized/gamma/Adam:0CGenerator/second/Generator/secondbatch_normalized/gamma/Adam/AssignCGenerator/second/Generator/secondbatch_normalized/gamma/Adam/read:02PGenerator/second/Generator/secondbatch_normalized/gamma/Adam/Initializer/zeros:0
ц
@Generator/second/Generator/secondbatch_normalized/gamma/Adam_1:0EGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/AssignEGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/read:02RGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/Initializer/zeros:0
ў
=Generator/second/Generator/secondbatch_normalized/beta/Adam:0BGenerator/second/Generator/secondbatch_normalized/beta/Adam/AssignBGenerator/second/Generator/secondbatch_normalized/beta/Adam/read:02OGenerator/second/Generator/secondbatch_normalized/beta/Adam/Initializer/zeros:0
а
?Generator/second/Generator/secondbatch_normalized/beta/Adam_1:0DGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/AssignDGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/read:02QGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/Initializer/zeros:0
ћ
<Generator/third/Generator/thirdfully_connected/kernel/Adam:0AGenerator/third/Generator/thirdfully_connected/kernel/Adam/AssignAGenerator/third/Generator/thirdfully_connected/kernel/Adam/read:02NGenerator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros:0
ю
>Generator/third/Generator/thirdfully_connected/kernel/Adam_1:0CGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/AssignCGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/read:02PGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros:0
ї
:Generator/third/Generator/thirdfully_connected/bias/Adam:0?Generator/third/Generator/thirdfully_connected/bias/Adam/Assign?Generator/third/Generator/thirdfully_connected/bias/Adam/read:02LGenerator/third/Generator/thirdfully_connected/bias/Adam/Initializer/zeros:0
ћ
<Generator/third/Generator/thirdfully_connected/bias/Adam_1:0AGenerator/third/Generator/thirdfully_connected/bias/Adam_1/AssignAGenerator/third/Generator/thirdfully_connected/bias/Adam_1/read:02NGenerator/third/Generator/thirdfully_connected/bias/Adam_1/Initializer/zeros:0
ћ
<Generator/third/Generator/thirdbatch_normalized/gamma/Adam:0AGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/AssignAGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/read:02NGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/Initializer/zeros:0
ю
>Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1:0CGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/AssignCGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/read:02PGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/Initializer/zeros:0
љ
;Generator/third/Generator/thirdbatch_normalized/beta/Adam:0@Generator/third/Generator/thirdbatch_normalized/beta/Adam/Assign@Generator/third/Generator/thirdbatch_normalized/beta/Adam/read:02MGenerator/third/Generator/thirdbatch_normalized/beta/Adam/Initializer/zeros:0
ў
=Generator/third/Generator/thirdbatch_normalized/beta/Adam_1:0BGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/AssignBGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/read:02OGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/Initializer/zeros:0
ћ
<Generator/forth/Generator/forthfully_connected/kernel/Adam:0AGenerator/forth/Generator/forthfully_connected/kernel/Adam/AssignAGenerator/forth/Generator/forthfully_connected/kernel/Adam/read:02NGenerator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros:0
ю
>Generator/forth/Generator/forthfully_connected/kernel/Adam_1:0CGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/AssignCGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/read:02PGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros:0
ї
:Generator/forth/Generator/forthfully_connected/bias/Adam:0?Generator/forth/Generator/forthfully_connected/bias/Adam/Assign?Generator/forth/Generator/forthfully_connected/bias/Adam/read:02LGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros:0
ћ
<Generator/forth/Generator/forthfully_connected/bias/Adam_1:0AGenerator/forth/Generator/forthfully_connected/bias/Adam_1/AssignAGenerator/forth/Generator/forthfully_connected/bias/Adam_1/read:02NGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros:0
ћ
<Generator/forth/Generator/forthbatch_normalized/gamma/Adam:0AGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/AssignAGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/read:02NGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros:0
ю
>Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1:0CGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/AssignCGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/read:02PGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros:0
љ
;Generator/forth/Generator/forthbatch_normalized/beta/Adam:0@Generator/forth/Generator/forthbatch_normalized/beta/Adam/Assign@Generator/forth/Generator/forthbatch_normalized/beta/Adam/read:02MGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros:0
ў
=Generator/forth/Generator/forthbatch_normalized/beta/Adam_1:0BGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/AssignBGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/read:02OGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros:0
ў
Generator/dense/kernel/Adam:0"Generator/dense/kernel/Adam/Assign"Generator/dense/kernel/Adam/read:02/Generator/dense/kernel/Adam/Initializer/zeros:0
а
Generator/dense/kernel/Adam_1:0$Generator/dense/kernel/Adam_1/Assign$Generator/dense/kernel/Adam_1/read:021Generator/dense/kernel/Adam_1/Initializer/zeros:0
љ
Generator/dense/bias/Adam:0 Generator/dense/bias/Adam/Assign Generator/dense/bias/Adam/read:02-Generator/dense/bias/Adam/Initializer/zeros:0
ў
Generator/dense/bias/Adam_1:0"Generator/dense/bias/Adam_1/Assign"Generator/dense/bias/Adam_1/read:02/Generator/dense/bias/Adam_1/Initializer/zeros:0
и║