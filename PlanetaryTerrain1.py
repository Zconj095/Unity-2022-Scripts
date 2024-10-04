class NoiseType:
    VALUE = "Value"
    VALUE_FRACTAL = "ValueFractal"
    PERLIN = "Perlin"
    PERLIN_FRACTAL = "PerlinFractal"
    SIMPLEX = "Simplex"
    SIMPLEX_FRACTAL = "SimplexFractal"
    CELLULAR = "Cellular"
    WHITE_NOISE = "WhiteNoise"
    CUBIC = "Cubic"
    CUBIC_FRACTAL = "CubicFractal"

class FractalType:
    FBM = "FBM"
    BILLOW = "Billow"
    RIGID_MULTI = "RigidMulti"

class Interp:
    LINEAR = "Linear"
    HERMITE = "Hermite"
    QUINTIC = "Quintic"

class CellularDistanceFunction:
    EUCLIDEAN = "Euclidean"
    MANHATTAN = "Manhattan"
    NATURAL = "Natural"

class CellularReturnType:
    CELL_VALUE = "CellValue"
    NOISE_LOOKUP = "NoiseLookup"
    DISTANCE = "Distance"
    DISTANCE2 = "Distance2"
    DISTANCE2_ADD = "Distance2Add"
    DISTANCE2_SUB = "Distance2Sub"
    DISTANCE2_MUL = "Distance2Mul"
    DISTANCE2_DIV = "Distance2Div"

class FastNoise:
    FN_INLINE = 256
    FN_CELLULAR_INDEX_MAX = 3

    def __init__(self, seed=1337):
        self.op_type = "Noise"
        self.m_seed = seed
        self.m_frequency = 0.01
        self.m_interp = Interp.QUINTIC
        self.m_noise_type = NoiseType.SIMPLEX
        self.m_octaves = 3
        self.m_lacunarity = 2.0
        self.m_gain = 0.5
        self.m_fractal_type = FractalType.FBM
        self.m_fractal_bounding = None
        self.m_cellular_distance_function = CellularDistanceFunction.EUCLIDEAN
        self.m_cellular_return_type = CellularReturnType.CELL_VALUE
        self.m_cellular_noise_lookup = None
        self.m_cellular_distance_index0 = 0
        self.m_cellular_distance_index1 = 1
        self.m_cellular_jitter = 0.45
        self.m_gradient_perturb_amp = 1.0
        self.calculate_fractal_bounding()

    @staticmethod
    def get_decimal_type():
        return 0.0

    def get_seed(self):
        return self.m_seed

    def set_seed(self, seed):
        self.m_seed = seed

    def set_frequency(self, frequency):
        self.m_frequency = frequency

    def set_interp(self, interp):
        self.m_interp = interp

    def set_noise_type(self, noise_type):
        self.m_noise_type = noise_type

    def set_fractal_octaves(self, octaves):
        self.m_octaves = octaves
        self.calculate_fractal_bounding()

    def set_fractal_lacunarity(self, lacunarity):
        self.m_lacunarity = lacunarity

    def set_fractal_gain(self, gain):
        self.m_gain = gain
        self.calculate_fractal_bounding()

    def set_fractal_type(self, fractal_type):
        self.m_fractal_type = fractal_type

    def set_cellular_distance_function(self, cellular_distance_function):
        self.m_cellular_distance_function = cellular_distance_function

    def set_cellular_return_type(self, cellular_return_type):
        self.m_cellular_return_type = cellular_return_type

    def set_cellular_distance2_indices(self, cellular_distance_index0, cellular_distance_index1):
        self.m_cellular_distance_index0 = min(cellular_distance_index0, cellular_distance_index1)
        self.m_cellular_distance_index1 = max(cellular_distance_index0, cellular_distance_index1)
        self.m_cellular_distance_index0 = min(max(self.m_cellular_distance_index0, 0), self.FN_CELLULAR_INDEX_MAX)
        self.m_cellular_distance_index1 = min(max(self.m_cellular_distance_index1, 0), self.FN_CELLULAR_INDEX_MAX)

    def set_cellular_jitter(self, cellular_jitter):
        self.m_cellular_jitter = cellular_jitter

    def set_cellular_noise_lookup(self, noise):
        self.m_cellular_noise_lookup = noise

    def set_gradient_perturb_amp(self, gradient_perturb_amp):
        self.m_gradient_perturb_amp = gradient_perturb_amp

    def calculate_fractal_bounding(self):
        # Calculation for fractal bounding (implementation needed)
        pass


import math

class Float2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Float3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

GRAD_2D = [
    Float2(-1, -1), Float2(1, -1), Float2(-1, 1), Float2(1, 1),
    Float2(0, -1), Float2(-1, 0), Float2(0, 1), Float2(1, 0),
]

GRAD_3D = [
    Float3(1, 1, 0), Float3(-1, 1, 0), Float3(1, -1, 0), Float3(-1, -1, 0),
    Float3(1, 0, 1), Float3(-1, 0, 1), Float3(1, 0, -1), Float3(-1, 0, -1),
    Float3(0, 1, 1), Float3(0, -1, 1), Float3(0, 1, -1), Float3(0, -1, -1),
    Float3(1, 1, 0), Float3(0, -1, 1), Float3(-1, 1, 0), Float3(0, -1, -1),
]

CELL_2D = [
    Float2(-0.2700222198, -0.9628540911), Float2(0.3863092627, -0.9223693152), Float2(0.04444859006, -0.999011673), Float2(-0.5992523158, -0.8005602176), 
    Float2(-0.7819280288, 0.6233687174), Float2(0.9464672271, 0.3227999196), Float2(-0.6514146797, -0.7587218957), Float2(0.9378472289, 0.347048376),
    Float2(-0.8497875957, -0.5271252623), Float2(-0.879042592, 0.4767432447), Float2(-0.892300288, -0.4514423508), Float2(-0.379844434, -0.9250503802), 
    Float2(-0.9951650832, 0.0982163789), Float2(0.7724397808, -0.6350880136), Float2(0.7573283322, -0.6530343002), Float2(-0.9928004525, -0.119780055),
    Float2(-0.0532665713, 0.9985803285), Float2(0.9754253726, -0.2203300762), Float2(-0.7665018163, 0.6422421394), Float2(0.991636706, 0.1290606184), 
    Float2(-0.994696838, 0.1028503788), Float2(-0.5379205513, -0.84299554), Float2(0.5022815471, -0.8647041387), Float2(0.4559821461, -0.8899889226),
    Float2(-0.8659131224, -0.5001944266), Float2(0.0879458407, -0.9961252577), Float2(-0.5051684983, 0.8630207346), Float2(0.7753185226, -0.6315704146), 
    Float2(-0.6921944612, 0.7217110418), Float2(-0.5191659449, -0.8546734591), Float2(0.8978622882, -0.4402764035), Float2(-0.1706774107, 0.9853269617),
    Float2(-0.9353430106, -0.3537420705), Float2(-0.9992404798, 0.03896746794), Float2(-0.2882064021, -0.9575683108), Float2(-0.9663811329, 0.2571137995), 
    Float2(-0.8759714238, -0.4823630009), Float2(-0.8303123018, -0.5572983775), Float2(0.05110133755, -0.9986934731), Float2(-0.8558373281, -0.5172450752),
    Float2(0.09887025282, 0.9951003332), Float2(0.9189016087, 0.3944867976), Float2(-0.2439375892, -0.9697909324), Float2(-0.8121409387, -0.5834613061), 
    Float2(-0.9910431363, 0.1335421355), Float2(0.8492423985, -0.5280031709), Float2(-0.9717838994, -0.2358729591), Float2(0.9949457207, 0.1004142068),
    Float2(0.6241065508, -0.7813392434), Float2(0.662910307, 0.7486988212), Float2(-0.7197418176, 0.6942418282), Float2(-0.8143370775, -0.5803922158), 
    Float2(0.104521054, -0.9945226741), Float2(-0.1065926113, -0.9943027784), Float2(0.445799684, -0.8951327509), Float2(0.105547406, 0.9944142724),
    Float2(-0.992790267, 0.1198644477), Float2(-0.8334366408, 0.552615025), Float2(0.9115561563, -0.4111755999), Float2(0.8285544909, -0.5599084351), 
    Float2(0.7217097654, -0.6921957921), Float2(0.4940492677, -0.8694339084), Float2(-0.3652321272, -0.9309164803), Float2(-0.9696606758, 0.2444548501),
    Float2(0.08925509731, -0.996008799), Float2(0.5354071276, -0.8445941083), Float2(-0.1053576186, 0.9944343981), Float2(-0.9890284586, 0.1477251101), 
    Float2(0.004856104961, 0.9999882091), Float2(0.9885598478, 0.1508291331), Float2(0.9286129562, -0.3710498316), Float2(-0.5832393863, -0.8123003252),
    Float2(0.3015207509, 0.9534596146), Float2(-0.9575110528, 0.2883965738), Float2(0.9715802154, -0.2367105511), Float2(0.229981792, 0.9731949318), 
    Float2(0.955763816, -0.2941352207), Float2(0.740956116, 0.6715534485), Float2(-0.9971513787, -0.07542630764), Float2(0.6905710663, -0.7232645452),
    Float2(-0.290713703, -0.9568100872), Float2(0.5912777791, -0.8064679708), Float2(-0.9454592212, -0.325740481), Float2(0.6664455681, 0.74555369), 
    Float2(0.6236134912, 0.7817328275), Float2(0.9126993851, -0.4086316587), Float2(-0.8191762011, 0.5735419353), Float2(-0.8812745759, -0.4726046147),
    Float2(0.9953313627, 0.09651672651), Float2(0.9855650846, -0.1692969699), Float2(-0.8495980887, 0.5274306472), Float2(0.6174853946, -0.7865823463), 
    Float2(0.8508156371, 0.52546432), Float2(0.9985032451, -0.05469249926), Float2(0.1971371563, -0.9803759185), Float2(0.6607855748, -0.7505747292),
    Float2(-0.03097494063, 0.9995201614), Float2(-0.6731660801, 0.739491331), Float2(-0.7195018362, -0.6944905383), Float2(0.9727511689, 0.2318515979), 
    Float2(0.9997059088, -0.0242506907), Float2(0.4421787429, -0.8969269532), Float2(0.9981350961, -0.061043673), Float2(-0.9173660799, -0.3980445648),
    Float2(-0.8150056635, -0.5794529907), Float2(-0.8789331304, 0.4769450202), Float2(0.0158605829, 0.999874213), Float2(-0.8095464474, 0.5870558317), 
    Float2(-0.9165898907, -0.3998286786), Float2(-0.8023542565, 0.5968480938), Float2(-0.5176737917, 0.8555780767), Float2(-0.8154407307, -0.5788405779),
    Float2(0.4022010347, -0.9155513791), Float2(-0.9052556868, -0.4248672045), Float2(0.7317445619, 0.6815789728), Float2(-0.5647632201, -0.8252529947), 
    Float2(-0.8403276335, -0.5420788397), Float2(-0.9314281527, 0.363925262), Float2(0.5238198472, 0.8518290719), Float2(0.7432803869, -0.6689800195),
    Float2(-0.985371561, -0.1704197369), Float2(0.4601468731, 0.88784281), Float2(0.825855404, 0.5638819483), Float2(0.6182366099, 0.7859920446), 
    Float2(0.8331502863, -0.553046653), Float2(0.1500307506, 0.9886813308), Float2(-0.662330369, -0.7492119075), Float2(-0.668598664, 0.743623444),
    Float2(0.7025606278, 0.7116238924), Float2(-0.5419389763, -0.8404178401), Float2(-0.3388616456, 0.9408362159), Float2(0.8331530315, 0.5530425174), 
    Float2(-0.2989720662, -0.9542618632), Float2(0.2638522993, 0.9645630949), Float2(0.124108739, -0.9922686234), Float2(-0.7282649308, -0.6852956957),
    Float2(0.6962500149, 0.7177993569), Float2(-0.9183535368, 0.3957610156), Float2(-0.6326102274, -0.7744703352), Float2(-0.9331891859, -0.359385508), 
    Float2(-0.1153779357, -0.9933216659), Float2(0.9514974788, -0.3076565421), Float2(-0.08987977445, -0.9959526224), Float2(0.6678496916, 0.7442961705),
    Float2(0.7952400393, -0.6062947138), Float2(-0.6462007402, -0.7631674805), Float2(-0.2733598753, 0.9619118351), Float2(0.9669590226, -0.254931851), 
    Float2(-0.9792894595, 0.2024651934), Float2(-0.5369502995, -0.8436138784), Float2(-0.270036471, -0.9628500944), Float2(-0.6400277131, 0.7683518247),
    Float2(-0.7854537493, -0.6189203566), Float2(0.06005905383, -0.9981948257), Float2(-0.02455770378, 0.9996984141), Float2(-0.65983623, 0.751409442), 
    Float2(-0.6253894466, -0.7803127835), Float2(-0.6210408851, -0.7837781695), Float2(0.8348888491, 0.5504185768), Float2(-0.1592275245, 0.9872419133),
    Float2(0.8367622488, 0.5475663786), Float2(-0.8675753916, -0.4973056806), Float2(-0.2022662628, -0.9793305667), Float2(0.9399189937, 0.3413975472), 
    Float2(0.9877404807, -0.1561049093), Float2(-0.9034455656, 0.4287028224), Float2(0.1269804218, -0.9919052235), Float2(-0.3819600854, 0.924178821),
    Float2(0.9754625894, 0.2201652486), Float2(-0.3204015856, -0.9472818081), Float2(-0.9874760884, 0.1577687387), Float2(0.02535348474, -0.9996785487), 
    Float2(0.4835130794, -0.8753371362), Float2(-0.2850799925, -0.9585037287), Float2(-0.06805516006, -0.99768156), Float2(-0.7885244045, -0.6150034663),
    Float2(0.3185392127, -0.9479096845), Float2(0.8880043089, 0.4598351306), Float2(0.6476921488, -0.7619021462), Float2(0.9820241299, 0.1887554194), 
    Float2(0.9357275128, -0.3527237187), Float2(-0.8894895414, 0.4569555293), Float2(0.7922791302, 0.6101588153), Float2(0.7483818261, 0.6632681526),
    Float2(-0.7288929755, -0.6846276581), Float2(0.8729032783, -0.4878932944), Float2(0.8288345784, 0.5594937369), Float2(0.08074567077, 0.9967347374), 
    Float2(0.9799148216, -0.1994165048), Float2(-0.580730673, -0.8140957471), Float2(-0.4700049791, -0.8826637636), Float2(0.2409492979, 0.9705377045),
    Float2(0.9437816757, -0.3305694308), Float2(-0.8927998638, -0.4504535528), Float2(-0.8069622304, 0.5906030467), Float2(0.06258973166, 0.9980393407), 
    Float2(-0.9312597469, 0.3643559849), Float2(0.5777449785, 0.8162173362), Float2(-0.3360095855, -0.941858566), Float2(0.697932075, -0.7161639607),
    Float2(-0.002008157227, -0.9999979837), Float2(-0.1827294312, -0.9831632392), Float2(-0.6523911722, 0.7578824173), Float2(-0.4302626911, -0.9027037258), 
    Float2(-0.9985126289, -0.05452091251), Float2(-0.01028102172, -0.9999471489), Float2(-0.4946071129, 0.8691166802), Float2(-0.2999350194, 0.9539596344),
    Float2(0.8165471961, 0.5772786819), Float2(0.2697460475, 0.962931498), Float2(-0.7306287391, -0.6827749597), Float2(-0.7590952064, -0.6509796216), 
    Float2(-0.907053853, 0.4210146171), Float2(-0.5104861064, -0.8598860013), Float2(0.8613350597, 0.5080373165), Float2(0.5007881595, -0.8655698812),
    Float2(-0.654158152, 0.7563577938), Float2(-0.8382755311, -0.545246856), Float2(0.6940070834, 0.7199681717), Float2(0.06950936031, 0.9975812994), 
    Float2(0.1702942185, -0.9853932612), Float2(0.2695973274, 0.9629731466), Float2(0.5519612192, -0.8338697815), Float2(0.225657487, -0.9742067022),
    Float2(0.4215262855, -0.9068161835), Float2(0.4881873305, -0.8727388672), Float2(-0.3683854996, -0.9296731273), Float2(-0.9825390578, 0.1860564427), 
    Float2(0.81256471, 0.5828709909), Float2(0.3196460933, -0.9475370046), Float2(0.9570913859, 0.2897862643), Float2(-0.6876655497, -0.7260276109),
    Float2(-0.9988770922, -0.047376731), Float2(-0.1250179027, 0.992154486), Float2(-0.8280133617, 0.560708367), Float2(0.9324863769, -0.3612051451), 
    Float2(0.6394653183, 0.7688199442), Float2(-0.01623847064, -0.9998681473), Float2(-0.9955014666, -0.09474613458), Float2(-0.81453315, 0.580117012),
    Float2(0.4037327978, -0.9148769469), Float2(0.9944263371, 0.1054336766), Float2(-0.1624711654, 0.9867132919), Float2(-0.9949487814, -0.100383875), 
    Float2(-0.6995302564, 0.7146029809), Float2(0.5263414922, -0.85027327), Float2(-0.5395221479, 0.841971408), Float2(0.6579370318, 0.7530729462),
    Float2(0.01426758847, -0.9998982128), Float2(-0.6734383991, 0.7392433447), Float2(0.639412098, -0.7688642071), Float2(0.9211571421, 0.3891908523), 
    Float2(-0.146637214, -0.9891903394), Float2(-0.782318098, 0.6228791163), Float2(-0.5039610839, -0.8637263605), Float2(-0.7743120191, -0.6328039957),
]

CELL_3D = [
    Float3(-0.7292736885, -0.6618439697, 0.1735581948), Float3(0.790292081, -0.5480887466, -0.2739291014), Float3(0.7217578935, 0.6226212466, -0.3023380997), 
    Float3(0.565683137, -0.8208298145, -0.0790000257), Float3(0.760049034, -0.5555979497, -0.3370999617), Float3(0.3713945616, 0.5011264475, 0.7816254623), 
    Float3(-0.1277062463, -0.4254438999, -0.8959289049), Float3(-0.2881560924, -0.5815838982, 0.7607405838),
    Float3(0.5849561111, -0.662820239, -0.4674352136), Float3(0.3307171178, 0.0391653737, 0.94291689), Float3(0.8712121778, -0.4113374369, -0.2679381538), 
    Float3(0.580981015, 0.7021915846, 0.4115677815), Float3(0.503756873, 0.6330056931, -0.5878203852), Float3(0.4493712205, 0.601390195, 0.6606022552), 
    Float3(-0.6878403724, 0.09018890807, -0.7202371714), Float3(-0.5958956522, -0.6469350577, 0.475797649),
    Float3(-0.5127052122, 0.1946921978, -0.8361987284), Float3(-0.9911507142, -0.05410276466, -0.1212153153), Float3(-0.2149721042, 0.9720882117, -0.09397607749), 
    Float3(-0.7518650936, -0.5428057603, 0.3742469607), Float3(0.5237068895, 0.8516377189, -0.02107817834), Float3(0.6333504779, 0.1926167129, -0.7495104896), 
    Float3(-0.06788241606, 0.3998305789, 0.9140719259), Float3(-0.5538628599, -0.4729896695, -0.6852128902),
    Float3(-0.7261455366, -0.5911990757, 0.3509933228), Float3(-0.9229274737, -0.1782808786, 0.3412049336), Float3(-0.6968815002, 0.6511274338, 0.3006480328), 
    Float3(0.9608044783, -0.2098363234, -0.1811724921), Float3(0.06817146062, -0.9743405129, 0.2145069156), Float3(-0.3577285196, -0.6697087264, -0.6507845481), 
    Float3(-0.1868621131, 0.7648617052, -0.6164974636), Float3(-0.6541697588, 0.3967914832, 0.6439087246),
    Float3(0.6993340405, -0.6164538506, 0.3618239211), Float3(-0.1546665739, 0.6291283928, 0.7617583057), Float3(-0.6841612949, -0.2580482182, -0.6821542638), 
    Float3(0.5383980957, 0.4258654885, 0.7271630328), Float3(-0.5026987823, -0.7939832935, -0.3418836993), Float3(0.3202971715, 0.2834415347, 0.9039195862), 
    Float3(0.8683227101, -0.0003762656404, -0.4959995258), Float3(0.791120031, -0.08511045745, 0.6057105799),
    Float3(-0.04011016052, -0.4397248749, 0.8972364289), Float3(0.9145119872, 0.3579346169, -0.1885487608), Float3(-0.9612039066, -0.2756484276, 0.01024666929), 
    Float3(0.6510361721, -0.2877799159, -0.7023778346), Float3(-0.2041786351, 0.7365237271, 0.644859585), Float3(-0.7718263711, 0.3790626912, 0.5104855816), 
    Float3(-0.3060082741, -0.7692987727, 0.5608371729), Float3(0.454007341, -0.5024843065, 0.7357899537),
    Float3(0.4816795475, 0.6021208291, -0.6367380315), Float3(0.6961980369, -0.3222197429, 0.641469197), Float3(-0.6532160499, -0.6781148932, 0.3368515753), 
    Float3(0.5089301236, -0.6154662304, -0.6018234363), Float3(-0.1635919754, -0.9133604627, -0.372840892), Float3(0.52408019, -0.8437664109, 0.1157505864), 
    Float3(0.5902587356, 0.4983817807, -0.6349883666), Float3(0.5863227872, 0.494764745, 0.6414307729),
    Float3(0.6779335087, 0.2341345225, 0.6968408593), Float3(0.7177054546, -0.6858979348, 0.120178631), Float3(-0.5328819713, -0.5205125012, 0.6671608058), 
    Float3(-0.8654874251, -0.0700727088, -0.4960053754), Float3(-0.2861810166, 0.7952089234, 0.5345495242), Float3(-0.04849529634, 0.9810836427, -0.1874115585), 
    Float3(-0.6358521667, 0.6058348682, 0.4781800233), Float3(0.6254794696, -0.2861619734, 0.7258696564),
    Float3(-0.2585259868, 0.5061949264, -0.8227581726), Float3(0.02136306781, 0.5064016808, -0.8620330371), Float3(0.200111773, 0.8599263484, 0.4695550591), 
    Float3(0.4743561372, 0.6014985084, -0.6427953014), Float3(0.6622993731, -0.5202474575, -0.5391679918), Float3(0.08084972818, -0.6532720452, 0.7527940996), 
    Float3(-0.6893687501, 0.0592860349, 0.7219805347), Float3(-0.1121887082, -0.9673185067, 0.2273952515),
    Float3(0.7344116094, 0.5979668656, -0.3210532909), Float3(0.5789393465, -0.2488849713, 0.7764570201), Float3(0.6988182827, 0.3557169806, -0.6205791146), 
    Float3(-0.8636845529, -0.2748771249, -0.4224826141), Float3(-0.4247027957, -0.4640880967, 0.777335046), Float3(0.5257722489, -0.8427017621, 0.1158329937), 
    Float3(0.9343830603, 0.316302472, -0.1639543925), Float3(-0.1016836419, -0.8057303073, -0.5834887393),
    Float3(-0.6529238969, 0.50602126, -0.5635892736), Float3(-0.2465286165, -0.9668205684, -0.06694497494), Float3(-0.9776897119, -0.2099250524, -0.007368825344), 
    Float3(0.7736893337, 0.5734244712, 0.2694238123), Float3(-0.6095087895, 0.4995678998, 0.6155736747), Float3(0.5794535482, 0.7434546771, 0.3339292269), 
    Float3(-0.8226211154, 0.08142581855, 0.5627293636), Float3(-0.510385483, 0.4703667658, 0.7199039967),
    Float3(-0.5764971849, -0.07231656274, -0.8138926898), Float3(0.7250628871, 0.3949971505, -0.5641463116), Float3(-0.1525424005, 0.4860840828, -0.8604958341), 
    Float3(-0.5550976208, -0.4957820792, 0.667882296), Float3(-0.1883614327, 0.9145869398, 0.357841725), Float3(0.7625556724, -0.5414408243, -0.3540489801), 
    Float3(-0.5870231946, -0.3226498013, -0.7424963803), Float3(0.3051124198, 0.2262544068, -0.9250488391),
    Float3(0.6379576059, 0.577242424, -0.5097070502), Float3(-0.5966775796, 0.1454852398, -0.7891830656), Float3(-0.658330573, 0.6555487542, -0.3699414651), 
    Float3(0.7434892426, 0.2351084581, 0.6260573129), Float3(0.5562114096, 0.8264360377, -0.0873632843), Float3(-0.3028940016, -0.8251527185, 0.4768419182), 
    Float3(0.1129343818, -0.985888439, -0.1235710781), Float3(0.5937652891, -0.5896813806, 0.5474656618),
    Float3(0.6757964092, -0.5835758614, -0.4502648413), Float3(0.7242302609, -0.1152719764, 0.6798550586), Float3(-0.9511914166, 0.0753623979, -0.2992580792), 
    Float3(0.2539470961, -0.1886339355, 0.9486454084), Float3(0.571433621, -0.1679450851, -0.8032795685), Float3(-0.06778234979, 0.3978269256, 0.9149531629), 
    Float3(0.6074972649, 0.733060024, -0.3058922593), Float3(-0.5435478392, 0.1675822484, 0.8224791405),
    Float3(-0.5876678086, -0.3380045064, -0.7351186982), Float3(-0.7967562402, 0.04097822706, -0.6029098428), Float3(-0.1996350917, 0.8706294745, 0.4496111079), 
    Float3(-0.02787660336, -0.9106232682, -0.4122962022), Float3(-0.7797625996, -0.6257634692, 0.01975775581), Float3(-0.5211232846, 0.7401644346, -0.4249554471), 
    Float3(0.8575424857, 0.4053272873, -0.3167501783), Float3(0.1045223322, 0.8390195772, -0.5339674439),
    Float3(0.3501822831, 0.9242524096, -0.1520850155), Float3(0.1987849858, 0.07647613266, 0.9770547224), Float3(0.7845996363, 0.6066256811, -0.1280964233), 
    Float3(0.09006737436, -0.9750989929, -0.2026569073), Float3(-0.8274343547, -0.542299559, 0.1458203587), Float3(-0.3485797732, -0.415802277, 0.840000362), 
    Float3(-0.2471778936, -0.7304819962, -0.6366310879), Float3(-0.3700154943, 0.8577948156, 0.3567584454),
    Float3(0.5913394901, -0.548311967, -0.5913303597), Float3(0.1204873514, -0.7626472379, -0.6354935001), Float3(0.616959265, 0.03079647928, 0.7863922953), 
    Float3(0.1258156836, -0.6640829889, -0.7369967419), Float3(-0.6477565124, -0.1740147258, -0.7417077429), Float3(0.6217889313, -0.7804430448, -0.06547655076), 
    Float3(0.6589943422, -0.6096987708, 0.4404473475), Float3(-0.2689837504, -0.6732403169, -0.6887635427),
    Float3(-0.3849775103, 0.5676542638, 0.7277093879), Float3(0.5754444408, 0.8110471154, -0.1051963504), Float3(0.9141593684, 0.3832947817, 0.131900567), 
    Float3(-0.107925319, 0.9245493968, 0.3654593525), Float3(0.377977089, 0.3043148782, 0.8743716458), Float3(-0.2142885215, -0.8259286236, 0.5214617324), 
    Float3(0.5802544474, 0.4148098596, -0.7008834116), Float3(-0.1982660881, 0.8567161266, -0.4761596756),
    Float3(-0.03381553704, 0.3773180787, -0.9254661404), Float3(-0.6867922841, -0.6656597827, 0.2919133642), Float3(0.7731742607, -0.2875793547, -0.5652430251), 
    Float3(-0.09655941928, 0.9193708367, -0.3813575004), Float3(0.2715702457, -0.9577909544, -0.09426605581), Float3(0.2451015704, -0.6917998565, -0.6792188003), 
    Float3(0.977700782, -0.1753855374, 0.1155036542), Float3(-0.5224739938, 0.8521606816, 0.02903615945),
    Float3(-0.7734880599, -0.5261292347, 0.3534179531), Float3(-0.7134492443, -0.269547243, 0.6467878011), Float3(0.1644037271, 0.5105846203, -0.8439637196), 
    Float3(0.6494635788, 0.05585611296, 0.7583384168), Float3(-0.4711970882, 0.5017280509, -0.7254255765), Float3(-0.6335764307, -0.2381686273, -0.7361091029), 
    Float3(-0.9021533097, -0.270947803, -0.3357181763), Float3(-0.3793711033, 0.872258117, 0.3086152025),
    Float3(-0.6855598966, -0.3250143309, 0.6514394162), Float3(0.2900942212, -0.7799057743, -0.5546100667), Float3(-0.2098319339, 0.85037073, 0.4825351604), 
    Float3(-0.4592603758, 0.6598504336, -0.5947077538), Float3(0.8715945488, 0.09616365406, -0.4807031248), Float3(-0.6776666319, 0.7118504878, -0.1844907016), 
    Float3(0.7044377633, 0.312427597, 0.637304036), Float3(-0.7052318886, -0.2401093292, -0.6670798253),
    Float3(0.081921007, -0.7207336136, -0.6883545647), Float3(-0.6993680906, -0.5875763221, -0.4069869034), Float3(-0.1281454481, 0.6419895885, 0.7559286424), 
    Float3(-0.6337388239, -0.6785471501, -0.3714146849), Float3(0.5565051903, -0.2168887573, -0.8020356851), Float3(-0.5791554484, 0.7244372011, -0.3738578718), 
    Float3(0.1175779076, -0.7096451073, 0.6946792478), Float3(-0.6134619607, 0.1323631078, 0.7785527795),
    Float3(0.6984635305, -0.02980516237, -0.715024719), Float3(0.8318082963, -0.3930171956, 0.3919597455), Float3(0.1469576422, 0.05541651717, -0.9875892167), 
    Float3(0.708868575, -0.2690503865, 0.6520101478), Float3(0.2726053183, 0.67369766, -0.68688995), Float3(-0.6591295371, 0.3035458599, -0.6880466294), 
    Float3(0.4815131379, -0.7528270071, 0.4487723203), Float3(0.9430009463, 0.1675647412, -0.2875261255),
    Float3(0.434802957, 0.7695304522, -0.4677277752), Float3(0.3931996188, 0.594473625, 0.7014236729), Float3(0.7254336655, -0.603925654, 0.3301814672), 
    Float3(0.7590235227, -0.6506083235, 0.02433313207), Float3(-0.8552768592, -0.3430042733, 0.3883935666), Float3(-0.6139746835, 0.6981725247, 0.3682257648), 
    Float3(-0.7465905486, -0.5752009504, 0.3342849376), Float3(0.5730065677, 0.810555537, -0.1210916791),
    Float3(-0.9225877367, -0.3475211012, -0.167514036), Float3(-0.7105816789, -0.4719692027, -0.5218416899), Float3(-0.08564609717, 0.3583001386, 0.929669703), 
    Float3(-0.8279697606, -0.2043157126, 0.5222271202), Float3(0.427944023, 0.278165994, 0.8599346446), Float3(0.5399079671, -0.7857120652, -0.3019204161), 
    Float3(0.5678404253, -0.5495413974, -0.6128307303), Float3(-0.9896071041, 0.1365639107, -0.04503418428),
    Float3(-0.6154342638, -0.6440875597, 0.4543037336), Float3(0.1074204368, -0.7946340692, 0.5975094525), Float3(-0.3595449969, -0.8885529948, 0.28495784), 
    Float3(-0.2180405296, 0.1529888965, 0.9638738118), Float3(-0.7277432317, -0.6164050508, -0.3007234646), Float3(0.7249729114, -0.00669719484, 0.6887448187), 
    Float3(-0.5553659455, -0.5336586252, 0.6377908264), Float3(0.5137558015, 0.7976208196, -0.3160000073),
    Float3(-0.3794024848, 0.9245608561, -0.03522751494), Float3(0.8229248658, 0.2745365933, -0.4974176556), Float3(-0.5404114394, 0.6091141441, 0.5804613989), 
    Float3(0.8036581901, -0.2703029469, 0.5301601931), Float3(0.6044318879, 0.6832968393, 0.4095943388), Float3(0.06389988817, 0.9658208605, -0.2512108074), 
    Float3(0.1087113286, 0.7402471173, -0.6634877936), Float3(-0.713427712, -0.6926784018, 0.1059128479),
    Float3(0.6458897819, -0.5724548511, -0.5050958653), Float3(-0.6553931414, 0.7381471625, 0.159995615), Float3(0.3910961323, 0.9188871375, -0.05186755998), 
    Float3(-0.4879022471, -0.5904376907, 0.6429111375), Float3(0.6014790094, 0.7707441366, -0.2101820095), Float3(-0.5677173047, 0.7511360995, 0.3368851762), 
    Float3(0.7858573506, 0.226674665, 0.5753666838), Float3(-0.4520345543, -0.604222686, -0.6561857263),
    Float3(0.002272116345, 0.4132844051, -0.9105991643), Float3(-0.5815751419, -0.5162925989, 0.6286591339), Float3(-0.03703704785, 0.8273785755, 0.5604221175), 
    Float3(-0.5119692504, 0.7953543429, -0.3244980058), Float3(-0.2682417366, -0.9572290247, -0.1084387619), Float3(-0.2322482736, -0.9679131102, -0.09594243324), 
    Float3(0.3554328906, -0.8881505545, 0.2913006227), Float3(0.7346520519, -0.4371373164, 0.5188422971),
    Float3(0.9985120116, 0.04659011161, -0.02833944577), Float3(-0.3727687496, -0.9082481361, 0.1900757285), Float3(0.91737377, -0.3483642108, 0.1925298489), 
    Float3(0.2714911074, 0.4147529736, -0.8684886582), Float3(0.5131763485, -0.7116334161, 0.4798207128), Float3(-0.8737353606, 0.18886992, -0.4482350644), 
    Float3(0.8460043821, -0.3725217914, 0.3814499973), Float3(0.8978727456, -0.1780209141, -0.4026575304),
    Float3(0.2178065647, -0.9698322841, -0.1094789531), Float3(-0.1518031304, -0.7788918132, -0.6085091231), Float3(-0.2600384876, -0.4755398075, -0.8403819825), 
    Float3(0.572313509, -0.7474340931, -0.3373418503), Float3(-0.7174141009, 0.1699017182, -0.6756111411), Float3(-0.684180784, 0.02145707593, -0.7289967412), 
    Float3(-0.2007447902, 0.06555605789, -0.9774476623), Float3(-0.1148803697, -0.8044887315, 0.5827524187),
    Float3(-0.7870349638, 0.03447489231, 0.6159443543), Float3(-0.2015596421, 0.6859872284, 0.6991389226), Float3(-0.08581082512, -0.10920836, -0.9903080513), 
    Float3(0.5532693395, 0.7325250401, -0.396610771), Float3(-0.1842489331, -0.9777375055, -0.1004076743), Float3(0.0775473789, -0.9111505856, 0.4047110257), 
    Float3(0.1399838409, 0.7601631212, -0.6344734459), Float3(0.4484419361, -0.845289248, 0.2904925424),
]

def fast_floor(f):
    return math.floor(f) if f >= 0 else math.floor(f) - 1

def fast_round(f):
    return round(f)

def lerp(a, b, t):
    return a + t * (b - a)

def interp_hermite_func(t):
    return t * t * (3 - 2 * t)

def interp_quintic_func(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

def cubic_lerp(a, b, c, d, t):
    p = (d - c) - (a - b)
    return t * t * t * p + t * t * ((a - b) - p) + t * (c - a) + b

class FastNoise:
    def calculate_fractal_bounding(self):
        amp = self.m_gain
        amp_fractal = 1
        for _ in range(1, self.m_octaves):
            amp_fractal += amp
            amp *= self.m_gain
        self.m_fractal_bounding = 1 / amp_fractal

    X_PRIME = 1619
    Y_PRIME = 31337
    Z_PRIME = 6971
    W_PRIME = 1013

    @staticmethod
    def hash_2d(seed, x, y):
        hash_val = seed
        hash_val ^= FastNoise.X_PRIME * x
        hash_val ^= FastNoise.Y_PRIME * y
        hash_val = hash_val * hash_val * hash_val * 60493
        hash_val = (hash_val >> 13) ^ hash_val
        return hash_val

    @staticmethod
    def hash_3d(seed, x, y, z):
        hash_val = seed
        hash_val ^= FastNoise.X_PRIME * x
        hash_val ^= FastNoise.Y_PRIME * y
        hash_val ^= FastNoise.Z_PRIME * z
        hash_val = hash_val * hash_val * hash_val * 60493
        hash_val = (hash_val >> 13) ^ hash_val
        return hash_val

    @staticmethod
    def hash_4d(seed, x, y, z, w):
        hash_val = seed
        hash_val ^= FastNoise.X_PRIME * x
        hash_val ^= FastNoise.Y_PRIME * y
        hash_val ^= FastNoise.Z_PRIME * z
        hash_val ^= FastNoise.W_PRIME * w
        hash_val = hash_val * hash_val * hash_val * 60493
        hash_val = (hash_val >> 13) ^ hash_val
        return hash_val

import math

class Float2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Float3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

X_PRIME = 1619
Y_PRIME = 31337
Z_PRIME = 6971
W_PRIME = 1013
FN_DECIMAL = float

GRAD_2D = [
    Float2(-1,-1), Float2(1,-1), Float2(-1, 1), Float2(1, 1),
    Float2(0,-1), Float2(-1, 0), Float2(0, 1), Float2(1, 0),
]

GRAD_3D = [
    Float3(1, 1, 0), Float3(-1, 1, 0), Float3(1,-1, 0), Float3(-1,-1, 0),
    Float3(1, 0, 1), Float3(-1, 0, 1), Float3(1, 0,-1), Float3(-1, 0,-1),
    Float3(0, 1, 1), Float3(0,-1, 1), Float3(0, 1,-1), Float3(0,-1,-1),
    Float3(1, 1, 0), Float3(0,-1, 1), Float3(-1, 1, 0), Float3(0,-1,-1),
]

def val_coord_2d(seed, x, y):
    n = seed
    n ^= X_PRIME * x
    n ^= Y_PRIME * y
    return (n * n * n * 60493) / 2147483648.0

def val_coord_3d(seed, x, y, z):
    n = seed
    n ^= X_PRIME * x
    n ^= Y_PRIME * y
    n ^= Z_PRIME * z
    return (n * n * n * 60493) / 2147483648.0

def val_coord_4d(seed, x, y, z, w):
    n = seed
    n ^= X_PRIME * x
    n ^= Y_PRIME * y
    n ^= Z_PRIME * z
    n ^= W_PRIME * w
    return (n * n * n * 60493) / 2147483648.0

def grad_coord_2d(seed, x, y, xd, yd):
    hash_val = seed
    hash_val ^= X_PRIME * x
    hash_val ^= Y_PRIME * y
    hash_val = hash_val * hash_val * hash_val * 60493
    hash_val = (hash_val >> 13) ^ hash_val
    g = GRAD_2D[hash_val & 7]
    return xd * g.x + yd * g.y

def grad_coord_3d(seed, x, y, z, xd, yd, zd):
    hash_val = seed
    hash_val ^= X_PRIME * x
    hash_val ^= Y_PRIME * y
    hash_val ^= Z_PRIME * z
    hash_val = hash_val * hash_val * hash_val * 60493
    hash_val = (hash_val >> 13) ^ hash_val
    g = GRAD_3D[hash_val & 15]
    return xd * g.x + yd * g.y + zd * g.z

def grad_coord_4d(seed, x, y, z, w, xd, yd, zd, wd):
    hash_val = seed
    hash_val ^= X_PRIME * x
    hash_val ^= Y_PRIME * y
    hash_val ^= Z_PRIME * z
    hash_val ^= W_PRIME * w
    hash_val = hash_val * hash_val * hash_val * 60493
    hash_val = (hash_val >> 13) ^ hash_val
    hash_val &= 31
    a, b, c = yd, zd, wd
    if hash_val >> 3 == 1:
        a, b, c = wd, xd, yd
    elif hash_val >> 3 == 2:
        a, b, c = zd, wd, xd
    elif hash_val >> 3 == 3:
        a, b, c = yd, zd, wd
    return ((-a if hash_val & 4 == 0 else a) +
            (-b if hash_val & 2 == 0 else b) +
            (-c if hash_val & 1 == 0 else c))

class FastNoise:
    def __init__(self, seed=1337):
        self.m_seed = seed
        self.m_frequency = 0.01
        self.m_interp = "Quintic"
        self.m_noiseType = "Simplex"
        self.m_octaves = 3
        self.m_lacunarity = 2.0
        self.m_gain = 0.5
        self.m_fractalType = "FBM"
        self.m_fractalBounding = 0
        self.m_cellularDistanceFunction = "Euclidean"
        self.m_cellularReturnType = "CellValue"
        self.m_cellularNoiseLookup = None
        self.m_cellularDistanceIndex0 = 0
        self.m_cellularDistanceIndex1 = 1
        self.m_cellularJitter = 0.45
        self.m_gradientPerturbAmp = 1.0
        self.calculate_fractal_bounding()

    def calculate_fractal_bounding(self):
        amp = self.m_gain
        amp_fractal = 1
        for _ in range(1, self.m_octaves):
            amp_fractal += amp
            amp *= self.m_gain
        self.m_fractalBounding = 1 / amp_fractal

    def get_noise(self, x, y, z=None):
        x *= self.m_frequency
        y *= self.m_frequency
        if z is not None:
            z *= self.m_frequency

        if z is None:
            return self._get_noise_2d(x, y)
        else:
            return self._get_noise_3d(x, y, z)

    def _get_noise_2d(self, x, y):
        if self.m_noiseType == "Value":
            return self.single_value(self.m_seed, x, y)
        elif self.m_noiseType == "ValueFractal":
            if self.m_fractalType == "FBM":
                return self.single_value_fractal_fbm(x, y)
            elif self.m_fractalType == "Billow":
                return self.single_value_fractal_billow(x, y)
            elif self.m_fractalType == "RigidMulti":
                return self.single_value_fractal_rigid_multi(x, y)
            else:
                return 0
        elif self.m_noiseType == "Perlin":
            return self.single_perlin(self.m_seed, x, y)
        elif self.m_noiseType == "PerlinFractal":
            if self.m_fractalType == "FBM":
                return self.single_perlin_fractal_fbm(x, y)
            elif self.m_fractalType == "Billow":
                return self.single_perlin_fractal_billow(x, y)
            elif self.m_fractalType == "RigidMulti":
                return self.single_perlin_fractal_rigid_multi(x, y)
            else:
                return 0
        elif self.m_noiseType == "Simplex":
            return self.single_simplex(self.m_seed, x, y)
        elif self.m_noiseType == "SimplexFractal":
            if self.m_fractalType == "FBM":
                return self.single_simplex_fractal_fbm(x, y)
            elif self.m_fractalType == "Billow":
                return self.single_simplex_fractal_billow(x, y)
            elif self.m_fractalType == "RigidMulti":
                return self.single_simplex_fractal_rigid_multi(x, y)
            else:
                return 0
        elif self.m_noiseType == "Cellular":
            if self.m_cellularReturnType in ["CellValue", "NoiseLookup", "Distance"]:
                return self.single_cellular(x, y)
            else:
                return self.single_cellular_2_edge(x, y)
        elif self.m_noiseType == "WhiteNoise":
            return self.get_white_noise(x, y)
        elif self.m_noiseType == "Cubic":
            return self.single_cubic(self.m_seed, x, y)
        elif self.m_noiseType == "CubicFractal":
            if self.m_fractalType == "FBM":
                return self.single_cubic_fractal_fbm(x, y)
            elif self.m_fractalType == "Billow":
                return self.single_cubic_fractal_billow(x, y)
            elif self.m_fractalType == "RigidMulti":
                return self.single_cubic_fractal_rigid_multi(x, y)
            else:
                return 0
        else:
            return 0

    def _get_noise_3d(self, x, y, z):
        if self.m_noiseType == "Value":
            return self.single_value(self.m_seed, x, y, z)
        elif self.m_noiseType == "ValueFractal":
            if self.m_fractalType == "FBM":
                return self.single_value_fractal_fbm(x, y, z)
            elif self.m_fractalType == "Billow":
                return self.single_value_fractal_billow(x, y, z)
            elif self.m_fractalType == "RigidMulti":
                return self.single_value_fractal_rigid_multi(x, y, z)
            else:
                return 0
        elif self.m_noiseType == "Perlin":
            return self.single_perlin(self.m_seed, x, y, z)
        elif self.m_noiseType == "PerlinFractal":
            if self.m_fractalType == "FBM":
                return self.single_perlin_fractal_fbm(x, y, z)
            elif self.m_fractalType == "Billow":
                return self.single_perlin_fractal_billow(x, y, z)
            elif self.m_fractalType == "RigidMulti":
                return self.single_perlin_fractal_rigid_multi(x, y, z)
            else:
                return 0
        elif self.m_noiseType == "Simplex":
            return self.single_simplex(self.m_seed, x, y, z)
        elif self.m_noiseType == "SimplexFractal":
            if self.m_fractalType == "FBM":
                return self.single_simplex_fractal_fbm(x, y, z)
            elif self.m_fractalType == "Billow":
                return self.single_simplex_fractal_billow(x, y, z)
            elif self.m_fractalType == "RigidMulti":
                return self.single_simplex_fractal_rigid_multi(x, y, z)
            else:
                return 0
        elif self.m_noiseType == "Cellular":
            if self.m_cellularReturnType in ["CellValue", "NoiseLookup", "Distance"]:
                return self.single_cellular(x, y, z)
            else:
                return self.single_cellular_2_edge(x, y, z)
        elif self.m_noiseType == "WhiteNoise":
            return self.get_white_noise(x, y, z)
        elif self.m_noiseType == "Cubic":
            return self.single_cubic(self.m_seed, x, y, z)
        elif self.m_noiseType == "CubicFractal":
            if self.m_fractalType == "FBM":
                return self.single_cubic_fractal_fbm(x, y, z)
            elif self.m_fractalType == "Billow":
                return self.single_cubic_fractal_billow(x, y, z)
            elif self.m_fractalType == "RigidMulti":
                return self.single_cubic_fractal_rigid_multi(x, y, z)
            else:
                return 0
        else:
            return 0

    # Placeholder methods for noise generation algorithms
    def single_value(self, seed, x, y, z=None):
        pass

    def single_value_fractal_fbm(self, x, y, z=None):
        pass

    def single_value_fractal_billow(self, x, y, z=None):
        pass

    def single_value_fractal_rigid_multi(self, x, y, z=None):
        pass

    def single_perlin(self, seed, x, y, z=None):
        pass

    def single_perlin_fractal_fbm(self, x, y, z=None):
        pass

    def single_perlin_fractal_billow(self, x, y, z=None):
        pass

    def single_perlin_fractal_rigid_multi(self, x, y, z=None):
        pass

    def single_simplex(self, seed, x, y, z=None):
        pass

    def single_simplex_fractal_fbm(self, x, y, z=None):
        pass

    def single_simplex_fractal_billow(self, x, y, z=None):
        pass

    def single_simplex_fractal_rigid_multi(self, x, y, z=None):
        pass

    def single_cellular(self, x, y, z=None):
        pass

    def single_cellular_2_edge(self, x, y, z=None):
        pass

    def get_white_noise(self, x, y, z=None):
        pass

    def single_cubic(self, seed, x, y, z=None):
        pass

    def single_cubic_fractal_fbm(self, x, y, z=None):
        pass

    def single_cubic_fractal_billow(self, x, y, z=None):
        pass

    def single_cubic_fractal_rigid_multi(self, x, y, z=None):
        pass

import math
import struct

class Float2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Float3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

X_PRIME = 1619
Y_PRIME = 31337
Z_PRIME = 6971
W_PRIME = 1013
FN_DECIMAL = float

GRAD_2D = [
    Float2(-1,-1), Float2(1,-1), Float2(-1, 1), Float2(1, 1),
    Float2(0,-1), Float2(-1, 0), Float2(0, 1), Float2(1, 0),
]

GRAD_3D = [
    Float3(1, 1, 0), Float3(-1, 1, 0), Float3(1,-1, 0), Float3(-1,-1, 0),
    Float3(1, 0, 1), Float3(-1, 0, 1), Float3(1, 0,-1), Float3(-1, 0,-1),
    Float3(0, 1, 1), Float3(0,-1, 1), Float3(0, 1,-1), Float3(0,-1,-1),
    Float3(1, 1, 0), Float3(0,-1, 1), Float3(-1, 1, 0), Float3(0,-1,-1),
]

def val_coord_2d(seed, x, y):
    n = seed
    n ^= X_PRIME * x
    n ^= Y_PRIME * y
    return (n * n * n * 60493) / 2147483648.0

def val_coord_3d(seed, x, y, z):
    n = seed
    n ^= X_PRIME * x
    n ^= Y_PRIME * y
    n ^= Z_PRIME * z
    return (n * n * n * 60493) / 2147483648.0

def val_coord_4d(seed, x, y, z, w):
    n = seed
    n ^= X_PRIME * x
    n ^= Y_PRIME * y
    n ^= Z_PRIME * z
    n ^= W_PRIME * w
    return (n * n * n * 60493) / 2147483648.0

def grad_coord_2d(seed, x, y, xd, yd):
    hash_val = seed
    hash_val ^= X_PRIME * x
    hash_val ^= Y_PRIME * y
    hash_val = hash_val * hash_val * hash_val * 60493
    hash_val = (hash_val >> 13) ^ hash_val
    g = GRAD_2D[hash_val & 7]
    return xd * g.x + yd * g.y

def grad_coord_3d(seed, x, y, z, xd, yd, zd):
    hash_val = seed
    hash_val ^= X_PRIME * x
    hash_val ^= Y_PRIME * y
    hash_val ^= Z_PRIME * z
    hash_val = hash_val * hash_val * hash_val * 60493
    hash_val = (hash_val >> 13) ^ hash_val
    g = GRAD_3D[hash_val & 15]
    return xd * g.x + yd * g.y + zd * g.z

def grad_coord_4d(seed, x, y, z, w, xd, yd, zd, wd):
    hash_val = seed
    hash_val ^= X_PRIME * x
    hash_val ^= Y_PRIME * y
    hash_val ^= Z_PRIME * z
    hash_val ^= W_PRIME * w
    hash_val = hash_val * hash_val * hash_val * 60493
    hash_val = (hash_val >> 13) ^ hash_val
    hash_val &= 31
    a, b, c = yd, zd, wd
    if hash_val >> 3 == 1:
        a, b, c = wd, xd, yd
    elif hash_val >> 3 == 2:
        a, b, c = zd, wd, xd
    elif hash_val >> 3 == 3:
        a, b, c = yd, zd, wd
    return ((-a if hash_val & 4 == 0 else a) +
            (-b if hash_val & 2 == 0 else b) +
            (-c if hash_val & 1 == 0 else c))

def float_cast_to_int(f):
    i = struct.unpack('!i', struct.pack('!f', f))[0]
    return i ^ (i >> 32)

class FastNoise:
    def __init__(self, seed=1337):
        self.m_seed = seed
        self.m_frequency = 0.01
        self.m_interp = "Quintic"
        self.m_noiseType = "Simplex"
        self.m_octaves = 3
        self.m_lacunarity = 2.0
        self.m_gain = 0.5
        self.m_fractalType = "FBM"
        self.m_fractalBounding = 0
        self.m_cellularDistanceFunction = "Euclidean"
        self.m_cellularReturnType = "CellValue"
        self.m_cellularNoiseLookup = None
        self.m_cellularDistanceIndex0 = 0
        self.m_cellularDistanceIndex1 = 1
        self.m_cellularJitter = 0.45
        self.m_gradientPerturbAmp = 1.0
        self.calculate_fractal_bounding()

    def calculate_fractal_bounding(self):
        amp = self.m_gain
        amp_fractal = 1
        for _ in range(1, self.m_octaves):
            amp_fractal += amp
            amp *= self.m_gain
        self.m_fractalBounding = 1 / amp_fractal

    def get_white_noise(self, x, y, z=None, w=None):
        if w is not None:
            xi = float_cast_to_int(x)
            yi = float_cast_to_int(y)
            zi = float_cast_to_int(z)
            wi = float_cast_to_int(w)
            return val_coord_4d(self.m_seed, xi, yi, zi, wi)
        elif z is not None:
            xi = float_cast_to_int(x)
            yi = float_cast_to_int(y)
            zi = float_cast_to_int(z)
            return val_coord_3d(self.m_seed, xi, yi, zi)
        else:
            xi = float_cast_to_int(x)
            yi = float_cast_to_int(y)
            return val_coord_2d(self.m_seed, xi, yi)

    def get_white_noise_int(self, x, y, z=None, w=None):
        if w is not None:
            return val_coord_4d(self.m_seed, x, y, z, w)
        elif z is not None:
            return val_coord_3d(self.m_seed, x, y, z)
        else:
            return val_coord_2d(self.m_seed, x, y)

    def get_value_fractal(self, x, y, z=None):
        x *= self.m_frequency
        y *= self.m_frequency
        if z is not None:
            z *= self.m_frequency
            if self.m_fractalType == "FBM":
                return self.single_value_fractal_fbm(x, y, z)
            elif self.m_fractalType == "Billow":
                return self.single_value_fractal_billow(x, y, z)
            elif self.m_fractalType == "RigidMulti":
                return self.single_value_fractal_rigid_multi(x, y, z)
        else:
            if self.m_fractalType == "FBM":
                return self.single_value_fractal_fbm(x, y)
            elif self.m_fractalType == "Billow":
                return self.single_value_fractal_billow(x, y)
            elif self.m_fractalType == "RigidMulti":
                return self.single_value_fractal_rigid_multi(x, y)
        return 0

    def single_value_fractal_fbm(self, x, y, z=None):
        seed = self.m_seed
        sum_value = self.single_value(seed, x, y, z)
        amp = 1

        for _ in range(1, self.m_octaves):
            x *= self.m_lacunarity
            y *= self.m_lacunarity
            if z is not None:
                z *= self.m_lacunarity

            amp *= self.m_gain
            sum_value += self.single_value(seed + 1, x, y, z) * amp

        return sum_value * self.m_fractalBounding

    def single_value_fractal_billow(self, x, y, z=None):
        seed = self.m_seed
        sum_value = abs(self.single_value(seed, x, y, z)) * 2 - 1
        amp = 1

        for _ in range(1, self.m_octaves):
            x *= self.m_lacunarity
            y *= self.m_lacunarity
            if z is not None:
                z *= self.m_lacunarity

            amp *= self.m_gain
            sum_value += (abs(self.single_value(seed + 1, x, y, z)) * 2 - 1) * amp

        return sum_value * self.m_fractalBounding

    def single_value_fractal_rigid_multi(self, x, y, z=None):
        seed = self.m_seed
        sum_value = 1 - abs(self.single_value(seed, x, y, z))
        amp = 1

        for _ in range(1, self.m_octaves):
            x *= self.m_lacunarity
            y *= self.m_lacunarity
            if z is not None:
                z *= self.m_lacunarity

            amp *= self.m_gain
            sum_value -= (1 - abs(self.single_value(seed + 1, x, y, z))) * amp

        return sum_value

    def get_value(self, x, y, z=None):
        if z is not None:
            return self.single_value(self.m_seed, x * self.m_frequency, y * self.m_frequency, z * self.m_frequency)
        else:
            return self.single_value(self.m_seed, x * self.m_frequency, y * self.m_frequency)

    def single_value(self, seed, x, y, z=None):
        x0 = math.floor(x)
        y0 = math.floor(y)
        if z is not None:
            z0 = math.floor(z)
            x1 = x0 + 1
            y1 = y0 + 1
            z1 = z0 + 1

            if self.m_interp == "Linear":
                xs = x - x0
                ys = y - y0
                zs = z - z0
            elif self.m_interp == "Hermite":
                xs = self.interp_hermite_func(x - x0)
                ys = self.interp_hermite_func(y - y0)
                zs = self.interp_hermite_func(z - z0)
            else:  # Quintic
                xs = self.interp_quintic_func(x - x0)
                ys = self.interp_quintic_func(y - y0)
                zs = self.interp_quintic_func(z - z0)

            xf00 = self.lerp(val_coord_3d(seed, x0, y0, z0), val_coord_3d(seed, x1, y0, z0), xs)
            xf10 = self.lerp(val_coord_3d(seed, x0, y1, z0), val_coord_3d(seed, x1, y1, z0), xs)
            xf01 = self.lerp(val_coord_3d(seed, x0, y0, z1), val_coord_3d(seed, x1, y0, z1), xs)
            xf11 = self.lerp(val_coord_3d(seed, x0, y1, z1), val_coord_3d(seed, x1, y1, z1), xs)

            yf0 = self.lerp(xf00, xf10, ys)
            yf1 = self.lerp(xf01, xf11, ys)

            return self.lerp(yf0, yf1, zs)
        else:
            x1 = x0 + 1
            y1 = y0 + 1

            if self.m_interp == "Linear":
                xs = x - x0
                ys = y - y0
            elif self.m_interp == "Hermite":
                xs = self.interp_hermite_func(x - x0)
                ys = self.interp_hermite_func(y - y0)
            else:  # Quintic
                xs = self.interp_quintic_func(x - x0)
                ys = self.interp_quintic_func(y - y0)

            xf0 = self.lerp(val_coord_2d(seed, x0, y0), val_coord_2d(seed, x1, y0), xs)
            xf1 = self.lerp(val_coord_2d(seed, x0, y1), val_coord_2d(seed, x1, y1), xs)

            return self.lerp(xf0, xf1, ys)

    @staticmethod
    def lerp(a, b, t):
        return a + t * (b - a)

    @staticmethod
    def interp_hermite_func(t):
        return t * t * (3 - 2 * t)

    @staticmethod
    def interp_quintic_func(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

import math
import struct

class Float2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Float3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

X_PRIME = 1619
Y_PRIME = 31337
Z_PRIME = 6971
W_PRIME = 1013
FN_DECIMAL = float

GRAD_2D = [
    Float2(-1,-1), Float2(1,-1), Float2(-1, 1), Float2(1, 1),
    Float2(0,-1), Float2(-1, 0), Float2(0, 1), Float2(1, 0),
]

GRAD_3D = [
    Float3(1, 1, 0), Float3(-1, 1, 0), Float3(1,-1, 0), Float3(-1,-1, 0),
    Float3(1, 0, 1), Float3(-1, 0, 1), Float3(1, 0,-1), Float3(-1, 0,-1),
    Float3(0, 1, 1), Float3(0,-1, 1), Float3(0, 1,-1), Float3(0,-1,-1),
    Float3(1, 1, 0), Float3(0,-1, 1), Float3(-1, 1, 0), Float3(0,-1,-1),
]

def val_coord_2d(seed, x, y):
    n = seed
    n ^= X_PRIME * x
    n ^= Y_PRIME * y
    return (n * n * n * 60493) / 2147483648.0

def val_coord_3d(seed, x, y, z):
    n = seed
    n ^= X_PRIME * x
    n ^= Y_PRIME * y
    n ^= Z_PRIME * z
    return (n * n * n * 60493) / 2147483648.0

def val_coord_4d(seed, x, y, z, w):
    n = seed
    n ^= X_PRIME * x
    n ^= Y_PRIME * y
    n ^= Z_PRIME * z
    n ^= W_PRIME * w
    return (n * n * n * 60493) / 2147483648.0

def grad_coord_2d(seed, x, y, xd, yd):
    hash_val = seed
    hash_val ^= X_PRIME * x
    hash_val ^= Y_PRIME * y
    hash_val = hash_val * hash_val * hash_val * 60493
    hash_val = (hash_val >> 13) ^ hash_val
    g = GRAD_2D[hash_val & 7]
    return xd * g.x + yd * g.y

def grad_coord_3d(seed, x, y, z, xd, yd, zd):
    hash_val = seed
    hash_val ^= X_PRIME * x
    hash_val ^= Y_PRIME * y
    hash_val ^= Z_PRIME * z
    hash_val = hash_val * hash_val * hash_val * 60493
    hash_val = (hash_val >> 13) ^ hash_val
    g = GRAD_3D[hash_val & 15]
    return xd * g.x + yd * g.y + zd * g.z

def grad_coord_4d(seed, x, y, z, w, xd, yd, zd, wd):
    hash_val = seed
    hash_val ^= X_PRIME * x
    hash_val ^= Y_PRIME * y
    hash_val ^= Z_PRIME * z
    hash_val ^= W_PRIME * w
    hash_val = hash_val * hash_val * hash_val * 60493
    hash_val = (hash_val >> 13) ^ hash_val
    hash_val &= 31
    a, b, c = yd, zd, wd
    if hash_val >> 3 == 1:
        a, b, c = wd, xd, yd
    elif hash_val >> 3 == 2:
        a, b, c = zd, wd, xd
    elif hash_val >> 3 == 3:
        a, b, c = yd, zd, wd
    return ((-a if hash_val & 4 == 0 else a) +
            (-b if hash_val & 2 == 0 else b) +
            (-c if hash_val & 1 == 0 else c))

def float_cast_to_int(f):
    i = struct.unpack('!i', struct.pack('!f', f))[0]
    return i ^ (i >> 32)

class FastNoise:
    def __init__(self, seed=1337):
        self.m_seed = seed
        self.m_frequency = 0.01
        self.m_interp = "Quintic"
        self.m_noiseType = "Simplex"
        self.m_octaves = 3
        self.m_lacunarity = 2.0
        self.m_gain = 0.5
        self.m_fractalType = "FBM"
        self.m_fractalBounding = 0
        self.m_cellularDistanceFunction = "Euclidean"
        self.m_cellularReturnType = "CellValue"
        self.m_cellularNoiseLookup = None
        self.m_cellularDistanceIndex0 = 0
        self.m_cellularDistanceIndex1 = 1
        self.m_cellularJitter = 0.45
        self.m_gradientPerturbAmp = 1.0
        self.calculate_fractal_bounding()

    def calculate_fractal_bounding(self):
        amp = self.m_gain
        amp_fractal = 1
        for _ in range(1, self.m_octaves):
            amp_fractal += amp
            amp *= self.m_gain
        self.m_fractalBounding = 1 / amp_fractal

    def get_white_noise(self, x, y, z=None, w=None):
        if w is not None:
            xi = float_cast_to_int(x)
            yi = float_cast_to_int(y)
            zi = float_cast_to_int(z)
            wi = float_cast_to_int(w)
            return val_coord_4d(self.m_seed, xi, yi, zi, wi)
        elif z is not None:
            xi = float_cast_to_int(x)
            yi = float_cast_to_int(y)
            zi = float_cast_to_int(z)
            return val_coord_3d(self.m_seed, xi, yi, zi)
        else:
            xi = float_cast_to_int(x)
            yi = float_cast_to_int(y)
            return val_coord_2d(self.m_seed, xi, yi)

    def get_white_noise_int(self, x, y, z=None, w=None):
        if w is not None:
            return val_coord_4d(self.m_seed, x, y, z, w)
        elif z is not None:
            return val_coord_3d(self.m_seed, x, y, z)
        else:
            return val_coord_2d(self.m_seed, x, y)

    def get_value_fractal(self, x, y, z=None):
        x *= self.m_frequency
        y *= self.m_frequency
        if z is not None:
            z *= self.m_frequency
            if self.m_fractalType == "FBM":
                return self.single_value_fractal_fbm(x, y, z)
            elif self.m_fractalType == "Billow":
                return self.single_value_fractal_billow(x, y, z)
            elif self.m_fractalType == "RigidMulti":
                return self.single_value_fractal_rigid_multi(x, y, z)
        else:
            if self.m_fractalType == "FBM":
                return self.single_value_fractal_fbm(x, y)
            elif self.m_fractalType == "Billow":
                return self.single_value_fractal_billow(x, y)
            elif self.m_fractalType == "RigidMulti":
                return self.single_value_fractal_rigid_multi(x, y)
        return 0

    def single_value_fractal_fbm(self, x, y, z=None):
        seed = self.m_seed
        sum_value = self.single_value(seed, x, y, z)
        amp = 1

        for _ in range(1, self.m_octaves):
            x *= self.m_lacunarity
            y *= self.m_lacunarity
            if z is not None:
                z *= self.m_lacunarity

            amp *= self.m_gain
            sum_value += self.single_value(seed + 1, x, y, z) * amp

        return sum_value * self.m_fractalBounding

    def single_value_fractal_billow(self, x, y, z=None):
        seed = self.m_seed
        sum_value = abs(self.single_value(seed, x, y, z)) * 2 - 1
        amp = 1

        for _ in range(1, self.m_octaves):
            x *= self.m_lacunarity
            y *= self.m_lacunarity
            if z is not None:
                z *= self.m_lacunarity

            amp *= self.m_gain
            sum_value += (abs(self.single_value(seed + 1, x, y, z)) * 2 - 1) * amp

        return sum_value * self.m_fractalBounding

    def single_value_fractal_rigid_multi(self, x, y, z=None):
        seed = self.m_seed
        sum_value = 1 - abs(self.single_value(seed, x, y, z))
        amp = 1

        for _ in range(1, self.m_octaves):
            x *= self.m_lacunarity
            y *= self.m_lacunarity
            if z is not None:
                z *= self.m_lacunarity

            amp *= self.m_gain
            sum_value -= (1 - abs(self.single_value(seed + 1, x, y, z))) * amp

        return sum_value

    def get_value(self, x, y, z=None):
        if z is not None:
            return self.single_value(self.m_seed, x * self.m_frequency, y * self.m_frequency, z * self.m_frequency)
        else:
            return self.single_value(self.m_seed, x * self.m_frequency, y * self.m_frequency)

    def single_value(self, seed, x, y, z=None):
        x0 = math.floor(x)
        y0 = math.floor(y)
        if z is not None:
            z0 = math.floor(z)
            x1 = x0 + 1
            y1 = y0 + 1
            z1 = z0 + 1

            if self.m_interp == "Linear":
                xs = x - x0
                ys = y - y0
                zs = z - z0
            elif self.m_interp == "Hermite":
                xs = self.interp_hermite_func(x - x0)
                ys = self.interp_hermite_func(y - y0)
                zs = self.interp_hermite_func(z - z0)
            else:  # Quintic
                xs = self.interp_quintic_func(x - x0)
                ys = self.interp_quintic_func(y - y0)
                zs = self.interp_quintic_func(z - z0)

            xf00 = self.lerp(val_coord_3d(seed, x0, y0, z0), val_coord_3d(seed, x1, y0, z0), xs)
            xf10 = self.lerp(val_coord_3d(seed, x0, y1, z0), val_coord_3d(seed, x1, y1, z0), xs)
            xf01 = self.lerp(val_coord_3d(seed, x0, y0, z1), val_coord_3d(seed, x1, y0, z1), xs)
            xf11 = self.lerp(val_coord_3d(seed, x0, y1, z1), val_coord_3d(seed, x1, y1, z1), xs)

            yf0 = self.lerp(xf00, xf10, ys)
            yf1 = self.lerp(xf01, xf11, ys)

            return self.lerp(yf0, yf1, zs)
        else:
            x1 = x0 + 1
            y1 = y0 + 1

            if self.m_interp == "Linear":
                xs = x - x0
                ys = y - y0
            elif self.m_interp == "Hermite":
                xs = self.interp_hermite_func(x - x0)
                ys = self.interp_hermite_func(y - y0)
            else:  # Quintic
                xs = self.interp_quintic_func(x - x0)
                ys = self.interp_quintic_func(y - y0)

            xf0 = self.lerp(val_coord_2d(seed, x0, y0), val_coord_2d(seed, x1, y0), xs)
            xf1 = self.lerp(val_coord_2d(seed, x0, y1), val_coord_2d(seed, x1, y1), xs)

            return self.lerp(xf0, xf1, ys)

    def get_perlin_fractal(self, x, y, z):
        x *= self.m_frequency
        y *= self.m_frequency
        z *= self.m_frequency

        if self.m_fractalType == "FBM":
            return self.single_perlin_fractal_fbm(x, y, z)
        elif self.m_fractalType == "Billow":
            return self.single_perlin_fractal_billow(x, y, z)
        elif self.m_fractalType == "RigidMulti":
            return self.single_perlin_fractal_rigid_multi(x, y, z)
        return 0

    def single_perlin_fractal_fbm(self, x, y, z):
        seed = self.m_seed
        sum_value = self.single_perlin(seed, x, y, z)
        amp = 1

        for _ in range(1, self.m_octaves):
            x *= self.m_lacunarity
            y *= self.m_lacunarity
            z *= self.m_lacunarity

            amp *= self.m_gain
            sum_value += self.single_perlin(seed + 1, x, y, z) * amp

        return sum_value * self.m_fractalBounding

    # Placeholder for single_perlin, single_perlin_fractal_billow, and single_perlin_fractal_rigid_multi methods
    def single_perlin(self, seed, x, y, z):
        pass

    def single_perlin_fractal_billow(self, x, y, z):
        pass

    def single_perlin_fractal_rigid_multi(self, x, y, z):
        pass

    @staticmethod
    def lerp(a, b, t):
        return a + t * (b - a)

    @staticmethod
    def interp_hermite_func(t):
        return t * t * (3 - 2 * t)

    @staticmethod
    def interp_quintic_func(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

import math
import struct

class Float2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Float3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

X_PRIME = 1619
Y_PRIME = 31337
Z_PRIME = 6971
W_PRIME = 1013
FN_DECIMAL = float

GRAD_2D = [
    Float2(-1,-1), Float2(1,-1), Float2(-1, 1), Float2(1, 1),
    Float2(0,-1), Float2(-1, 0), Float2(0, 1), Float2(1, 0),
]

GRAD_3D = [
    Float3(1, 1, 0), Float3(-1, 1, 0), Float3(1,-1, 0), Float3(-1,-1, 0),
    Float3(1, 0, 1), Float3(-1, 0, 1), Float3(1, 0,-1), Float3(-1, 0,-1),
    Float3(0, 1, 1), Float3(0,-1, 1), Float3(0, 1,-1), Float3(0,-1,-1),
    Float3(1, 1, 0), Float3(0,-1, 1), Float3(-1, 1, 0), Float3(0,-1,-1),
]

def val_coord_2d(seed, x, y):
    n = seed
    n ^= X_PRIME * x
    n ^= Y_PRIME * y
    return (n * n * n * 60493) / 2147483648.0

def val_coord_3d(seed, x, y, z):
    n = seed
    n ^= X_PRIME * x
    n ^= Y_PRIME * y
    n ^= Z_PRIME * z
    return (n * n * n * 60493) / 2147483648.0

def val_coord_4d(seed, x, y, z, w):
    n = seed
    n ^= X_PRIME * x
    n ^= Y_PRIME * y
    n ^= Z_PRIME * z
    n ^= W_PRIME * w
    return (n * n * n * 60493) / 2147483648.0

def grad_coord_3d(seed, x, y, z, xd, yd, zd):
    hash_val = seed
    hash_val ^= X_PRIME * x
    hash_val ^= Y_PRIME * y
    hash_val ^= Z_PRIME * z
    hash_val = hash_val * hash_val * hash_val * 60493
    hash_val = (hash_val >> 13) ^ hash_val
    g = GRAD_3D[hash_val & 15]
    return xd * g.x + yd * g.y + zd * g.z

def float_cast_to_int(f):
    i = struct.unpack('!i', struct.pack('!f', f))[0]
    return i ^ (i >> 32)

class FastNoise:
    def __init__(self, seed=1337):
        self.m_seed = seed
        self.m_frequency = 0.01
        self.m_interp = "Quintic"
        self.m_noiseType = "Simplex"
        self.m_octaves = 3
        self.m_lacunarity = 2.0
        self.m_gain = 0.5
        self.m_fractalType = "FBM"
        self.m_fractalBounding = 0
        self.m_cellularDistanceFunction = "Euclidean"
        self.m_cellularReturnType = "CellValue"
        self.m_cellularNoiseLookup = None
        self.m_cellularDistanceIndex0 = 0
        self.m_cellularDistanceIndex1 = 1
        self.m_cellularJitter = 0.45
        self.m_gradientPerturbAmp = 1.0
        self.calculate_fractal_bounding()

    def calculate_fractal_bounding(self):
        amp = self.m_gain
        amp_fractal = 1
        for _ in range(1, self.m_octaves):
            amp_fractal += amp
            amp *= self.m_gain
        self.m_fractalBounding = 1 / amp_fractal

    def get_white_noise(self, x, y, z=None, w=None):
        if w is not None:
            xi = float_cast_to_int(x)
            yi = float_cast_to_int(y)
            zi = float_cast_to_int(z)
            wi = float_cast_to_int(w)
            return val_coord_4d(self.m_seed, xi, yi, zi, wi)
        elif z is not None:
            xi = float_cast_to_int(x)
            yi = float_cast_to_int(y)
            zi = float_cast_to_int(z)
            return val_coord_3d(self.m_seed, xi, yi, zi)
        else:
            xi = float_cast_to_int(x)
            yi = float_cast_to_int(y)
            return val_coord_2d(self.m_seed, xi, yi)

    def get_white_noise_int(self, x, y, z=None, w=None):
        if w is not None:
            return val_coord_4d(self.m_seed, x, y, z, w)
        elif z is not None:
            return val_coord_3d(self.m_seed, x, y, z)
        else:
            return val_coord_2d(self.m_seed, x, y)

    def get_value_fractal(self, x, y, z=None):
        x *= self.m_frequency
        y *= self.m_frequency
        if z is not None:
            z *= self.m_frequency
            if self.m_fractalType == "FBM":
                return self.single_value_fractal_fbm(x, y, z)
            elif self.m_fractalType == "Billow":
                return self.single_value_fractal_billow(x, y, z)
            elif self.m_fractalType == "RigidMulti":
                return self.single_value_fractal_rigid_multi(x, y, z)
        else:
            if self.m_fractalType == "FBM":
                return self.single_value_fractal_fbm(x, y)
            elif self.m_fractalType == "Billow":
                return self.single_value_fractal_billow(x, y)
            elif self.m_fractalType == "RigidMulti":
                return self.single_value_fractal_rigid_multi(x, y)
        return 0

    def single_value_fractal_fbm(self, x, y, z=None):
        seed = self.m_seed
        sum_value = self.single_value(seed, x, y, z)
        amp = 1

        for _ in range(1, self.m_octaves):
            x *= self.m_lacunarity
            y *= self.m_lacunarity
            if z is not None:
                z *= self.m_lacunarity

            amp *= self.m_gain
            sum_value += self.single_value(seed + 1, x, y, z) * amp

        return sum_value * self.m_fractalBounding

    def single_value_fractal_billow(self, x, y, z=None):
        seed = self.m_seed
        sum_value = abs(self.single_value(seed, x, y, z)) * 2 - 1
        amp = 1

        for _ in range(1, self.m_octaves):
            x *= self.m_lacunarity
            y *= self.m_lacunarity
            if z is not None:
                z *= self.m_lacunarity

            amp *= self.m_gain
            sum_value += (abs(self.single_value(seed + 1, x, y, z)) * 2 - 1) * amp

        return sum_value * self.m_fractalBounding

    def single_value_fractal_rigid_multi(self, x, y, z=None):
        seed = self.m_seed
        sum_value = 1 - abs(self.single_value(seed, x, y, z))
        amp = 1

        for _ in range(1, self.m_octaves):
            x *= self.m_lacunarity
            y *= self.m_lacunarity
            if z is not None:
                z *= self.m_lacunarity

            amp *= self.m_gain
            sum_value -= (1 - abs(self.single_value(seed + 1, x, y, z))) * amp

        return sum_value

    def get_value(self, x, y, z=None):
        if z is not None:
            return self.single_value(self.m_seed, x * self.m_frequency, y * self.m_frequency, z * self.m_frequency)
        else:
            return self.single_value(self.m_seed, x * self.m_frequency, y * self.m_frequency)

    def single_value(self, seed, x, y, z=None):
        x0 = math.floor(x)
        y0 = math.floor(y)
        if z is not None:
            z0 = math.floor(z)
            x1 = x0 + 1
            y1 = y0 + 1
            z1 = z0 + 1

            if self.m_interp == "Linear":
                xs = x - x0
                ys = y - y0
                zs = z - z0
            elif self.m_interp == "Hermite":
                xs = self.interp_hermite_func(x - x0)
                ys = self.interp_hermite_func(y - y0)
                zs = self.interp_hermite_func(z - z0)
            else:  # Quintic
                xs = self.interp_quintic_func(x - x0)
                ys = self.interp_quintic_func(y - y0)
                zs = self.interp_quintic_func(z - z0)

            xf00 = self.lerp(val_coord_3d(seed, x0, y0, z0), val_coord_3d(seed, x1, y0, z0), xs)
            xf10 = self.lerp(val_coord_3d(seed, x0, y1, z0), val_coord_3d(seed, x1, y1, z0), xs)
            xf01 = self.lerp(val_coord_3d(seed, x0, y0, z1), val_coord_3d(seed, x1, y0, z1), xs)
            xf11 = self.lerp(val_coord_3d(seed, x0, y1, z1), val_coord_3d(seed, x1, y1, z1), xs)

            yf0 = self.lerp(xf00, xf10, ys)
            yf1 = self.lerp(xf01, xf11, ys)

            return self.lerp(yf0, yf1, zs)
        else:
            x1 = x0 + 1
            y1 = y0 + 1

            if self.m_interp == "Linear":
                xs = x - x0
                ys = y - y0
            elif self.m_interp == "Hermite":
                xs = self.interp_hermite_func(x - x0)
                ys = self.interp_hermite_func(y - y0)
            else:  # Quintic
                xs = self.interp_quintic_func(x - x0)
                ys = self.interp_quintic_func(y - y0)

            xf0 = self.lerp(val_coord_2d(seed, x0, y0), val_coord_2d(seed, x1, y0), xs)
            xf1 = self.lerp(val_coord_2d(seed, x0, y1), val_coord_2d(seed, x1, y1), xs)

            return self.lerp(xf0, xf1, ys)

    def get_perlin_fractal(self, x, y, z):
        x *= self.m_frequency
        y *= self.m_frequency
        z *= self.m_frequency

        if self.m_fractalType == "FBM":
            return self.single_perlin_fractal_fbm(x, y, z)
        elif self.m_fractalType == "Billow":
            return self.single_perlin_fractal_billow(x, y, z)
        elif self.m_fractalType == "RigidMulti":
            return self.single_perlin_fractal_rigid_multi(x, y, z)
        return 0

    def single_perlin_fractal_fbm(self, x, y, z):
        seed = self.m_seed
        sum_value = self.single_perlin(seed, x, y, z)
        amp = 1

        for _ in range(1, self.m_octaves):
            x *= self.m_lacunarity
            y *= self.m_lacunarity
            z *= self.m_lacunarity

            amp *= self.m_gain
            sum_value += self.single_perlin(seed + 1, x, y, z) * amp

        return sum_value * self.m_fractalBounding

    def single_perlin_fractal_billow(self, x, y, z):
        seed = self.m_seed
        sum_value = abs(self.single_perlin(seed, x, y, z)) * 2 - 1
        amp = 1

        for _ in range(1, self.m_octaves):
            x *= self.m_lacunarity
            y *= self.m_lacunarity
            z *= self.m_lacunarity

            amp *= self.m_gain
            sum_value += (abs(self.single_perlin(seed + 1, x, y, z)) * 2 - 1) * amp

        return sum_value * self.m_fractalBounding

    def single_perlin_fractal_rigid_multi(self, x, y, z):
        seed = self.m_seed
        sum_value = 1 - abs(self.single_perlin(seed, x, y, z))
        amp = 1

        for _ in range(1, self.m_octaves):
            x *= self.m_lacunarity
            y *= self.m_lacunarity
            z *= self.m_lacunarity

            amp *= self.m_gain
            sum_value -= (1 - abs(self.single_perlin(seed + 1, x, y, z))) * amp

        return sum_value

    def get_perlin(self, x, y, z):
        return self.single_perlin(self.m_seed, x * self.m_frequency, y * self.m_frequency, z * self.m_frequency)

    def single_perlin(self, seed, x, y, z):
        x0 = math.floor(x)
        y0 = math.floor(y)
        z0 = math.floor(z)
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        if self.m_interp == "Linear":
            xs = x - x0
            ys = y - y0
            zs = z - z0
        elif self.m_interp == "Hermite":
            xs = self.interp_hermite_func(x - x0)
            ys = self.interp_hermite_func(y - y0)
            zs = self.interp_hermite_func(z - z0)
        else:  # Quintic
            xs = self.interp_quintic_func(x - x0)
            ys = self.interp_quintic_func(y - y0)
            zs = self.interp_quintic_func(z - z0)

        xd0 = x - x0
        yd0 = y - y0
        zd0 = z - z0
        xd1 = xd0 - 1
        yd1 = yd0 - 1
        zd1 = zd0 - 1

        xf00 = self.lerp(grad_coord_3d(seed, x0, y0, z0, xd0, yd0, zd0), grad_coord_3d(seed, x1, y0, z0, xd1, yd0, zd0), xs)
        xf10 = self.lerp(grad_coord_3d(seed, x0, y1, z0, xd0, yd1, zd0), grad_coord_3d(seed, x1, y1, z0, xd1, yd1, zd0), xs)
        xf01 = self.lerp(grad_coord_3d(seed, x0, y0, z1, xd0, yd0, zd1), grad_coord_3d(seed, x1, y0, z1, xd1, yd0, zd1), xs)
        xf11 = self.lerp(grad_coord_3d(seed, x0, y1, z1, xd0, yd1, zd1), grad_coord_3d(seed, x1, y1, z1, xd1, yd1, zd1), xs)

        yf0 = self.lerp(xf00, xf10, ys)
        yf1 = self.lerp(xf01, xf11, ys)

        return self.lerp(yf0, yf1, zs)

    @staticmethod
    def lerp(a, b, t):
        return a + t * (b - a)

    @staticmethod
    def interp_hermite_func(t):
        return t * t * (3 - 2 * t)

    @staticmethod
    def interp_quintic_func(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

import math
import struct

class Float2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Float3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

X_PRIME = 1619
Y_PRIME = 31337
Z_PRIME = 6971
W_PRIME = 1013
FN_DECIMAL = float

GRAD_2D = [
    Float2(-1,-1), Float2(1,-1), Float2(-1, 1), Float2(1, 1),
    Float2(0,-1), Float2(-1, 0), Float2(0, 1), Float2(1, 0),
]

GRAD_3D = [
    Float3(1, 1, 0), Float3(-1, 1, 0), Float3(1,-1, 0), Float3(-1,-1, 0),
    Float3(1, 0, 1), Float3(-1, 0, 1), Float3(1, 0,-1), Float3(-1, 0,-1),
    Float3(0, 1, 1), Float3(0,-1, 1), Float3(0, 1,-1), Float3(0,-1,-1),
    Float3(1, 1, 0), Float3(0,-1, 1), Float3(-1, 1, 0), Float3(0,-1,-1),
]

def val_coord_2d(seed, x, y):
    n = seed
    n ^= X_PRIME * x
    n ^= Y_PRIME * y
    return (n * n * n * 60493) / 2147483648.0

def val_coord_3d(seed, x, y, z):
    n = seed
    n ^= X_PRIME * x
    n ^= Y_PRIME * y
    n ^= Z_PRIME * z
    return (n * n * n * 60493) / 2147483648.0

def val_coord_4d(seed, x, y, z, w):
    n = seed
    n ^= X_PRIME * x
    n ^= Y_PRIME * y
    n ^= Z_PRIME * z
    n ^= W_PRIME * w
    return (n * n * n * 60493) / 2147483648.0

def grad_coord_2d(seed, x, y, xd, yd):
    hash_val = seed
    hash_val ^= X_PRIME * x
    hash_val ^= Y_PRIME * y
    hash_val = hash_val * hash_val * hash_val * 60493
    hash_val = (hash_val >> 13) ^ hash_val
    g = GRAD_2D[hash_val & 7]
    return xd * g.x + yd * g.y

def grad_coord_3d(seed, x, y, z, xd, yd, zd):
    hash_val = seed
    hash_val ^= X_PRIME * x
    hash_val ^= Y_PRIME * y
    hash_val ^= Z_PRIME * z
    hash_val = hash_val * hash_val * hash_val * 60493
    hash_val = (hash_val >> 13) ^ hash_val
    g = GRAD_3D[hash_val & 15]
    return xd * g.x + yd * g.y + zd * g.z

def float_cast_to_int(f):
    i = struct.unpack('!i', struct.pack('!f', f))[0]
    return i ^ (i >> 32)

class FastNoise:
    def __init__(self, seed=1337):
        self.m_seed = seed
        self.m_frequency = 0.01
        self.m_interp = "Quintic"
        self.m_noiseType = "Simplex"
        self.m_octaves = 3
        self.m_lacunarity = 2.0
        self.m_gain = 0.5
        self.m_fractalType = "FBM"
        self.m_fractalBounding = 0
        self.m_cellularDistanceFunction = "Euclidean"
        self.m_cellularReturnType = "CellValue"
        self.m_cellularNoiseLookup = None
        self.m_cellularDistanceIndex0 = 0
        self.m_cellularDistanceIndex1 = 1
        self.m_cellularJitter = 0.45
        self.m_gradientPerturbAmp = 1.0
        self.calculate_fractal_bounding()

    def calculate_fractal_bounding(self):
        amp = self.m_gain
        amp_fractal = 1
        for _ in range(1, self.m_octaves):
            amp_fractal += amp
            amp *= self.m_gain
        self.m_fractalBounding = 1 / amp_fractal

    def get_white_noise(self, x, y, z=None, w=None):
        if w is not None:
            xi = float_cast_to_int(x)
            yi = float_cast_to_int(y)
            zi = float_cast_to_int(z)
            wi = float_cast_to_int(w)
            return val_coord_4d(self.m_seed, xi, yi, zi, wi)
        elif z is not None:
            xi = float_cast_to_int(x)
            yi = float_cast_to_int(y)
            zi = float_cast_to_int(z)
            return val_coord_3d(self.m_seed, xi, yi, zi)
        else:
            xi = float_cast_to_int(x)
            yi = float_cast_to_int(y)
            return val_coord_2d(self.m_seed, xi, yi)

    def get_white_noise_int(self, x, y, z=None, w=None):
        if w is not None:
            return val_coord_4d(self.m_seed, x, y, z, w)
        elif z is not None:
            return val_coord_3d(self.m_seed, x, y, z)
        else:
            return val_coord_2d(self.m_seed, x, y)

    def get_value_fractal(self, x, y, z=None):
        x *= self.m_frequency
        y *= self.m_frequency
        if z is not None:
            z *= self.m_frequency
            if self.m_fractalType == "FBM":
                return self.single_value_fractal_fbm(x, y, z)
            elif self.m_fractalType == "Billow":
                return self.single_value_fractal_billow(x, y, z)
            elif self.m_fractalType == "RigidMulti":
                return self.single_value_fractal_rigid_multi(x, y, z)
        else:
            if self.m_fractalType == "FBM":
                return self.single_value_fractal_fbm(x, y)
            elif self.m_fractalType == "Billow":
                return self.single_value_fractal_billow(x, y)
            elif self.m_fractalType == "RigidMulti":
                return self.single_value_fractal_rigid_multi(x, y)
        return 0

    def single_value_fractal_fbm(self, x, y, z=None):
        seed = self.m_seed
        sum_value = self.single_value(seed, x, y, z)
        amp = 1

        for _ in range(1, self.m_octaves):
            x *= self.m_lacunarity
            y *= self.m_lacunarity
            if z is not None:
                z *= self.m_lacunarity

            amp *= self.m_gain
            sum_value += self.single_value(seed + 1, x, y, z) * amp

        return sum_value * self.m_fractalBounding

    def single_value_fractal_billow(self, x, y, z=None):
        seed = self.m_seed
        sum_value = abs(self.single_value(seed, x, y, z)) * 2 - 1
        amp = 1

        for _ in range(1, self.m_octaves):
            x *= self.m_lacunarity
            y *= self.m_lacunarity
            if z is not None:
                z *= self.m_lacunarity

            amp *= self.m_gain
            sum_value += (abs(self.single_value(seed + 1, x, y, z)) * 2 - 1) * amp

        return sum_value * self.m_fractalBounding

    def single_value_fractal_rigid_multi(self, x, y, z=None):
        seed = self.m_seed
        sum_value = 1 - abs(self.single_value(seed, x, y, z))
        amp = 1

        for _ in range(1, self.m_octaves):
            x *= self.m_lacunarity
            y *= self.m_lacunarity
            if z is not None:
                z *= self.m_lacunarity

            amp *= self.m_gain
            sum_value -= (1 - abs(self.single_value(seed + 1, x, y, z))) * amp

        return sum_value

    def get_value(self, x, y, z=None):
        if z is not None:
            return self.single_value(self.m_seed, x * self.m_frequency, y * self.m_frequency, z * self.m_frequency)
        else:
            return self.single_value(self.m_seed, x * self.m_frequency, y * self.m_frequency)

    def single_value(self, seed, x, y, z=None):
        x0 = math.floor(x)
        y0 = math.floor(y)
        if z is not None:
            z0 = math.floor(z)
            x1 = x0 + 1
            y1 = y0 + 1
            z1 = z0 + 1

            if self.m_interp == "Linear":
                xs = x - x0
                ys = y - y0
                zs = z - z0
            elif self.m_interp == "Hermite":
                xs = self.interp_hermite_func(x - x0)
                ys = self.interp_hermite_func(y - y0)
                zs = self.interp_hermite_func(z - z0)
            else:  # Quintic
                xs = self.interp_quintic_func(x - x0)
                ys = self.interp_quintic_func(y - y0)
                zs = self.interp_quintic_func(z - z0)

            xf00 = self.lerp(val_coord_3d(seed, x0, y0, z0), val_coord_3d(seed, x1, y0, z0), xs)
            xf10 = self.lerp(val_coord_3d(seed, x0, y1, z0), val_coord_3d(seed, x1, y1, z0), xs)
            xf01 = self.lerp(val_coord_3d(seed, x0, y0, z1), val_coord_3d(seed, x1, y0, z1), xs)
            xf11 = self.lerp(val_coord_3d(seed, x0, y1, z1), val_coord_3d(seed, x1, y1, z1), xs)

            yf0 = self.lerp(xf00, xf10, ys)
            yf1 = self.lerp(xf01, xf11, ys)

            return self.lerp(yf0, yf1, zs)
        else:
            x1 = x0 + 1
            y1 = y0 + 1

            if self.m_interp == "Linear":
                xs = x - x0
                ys = y - y0
            elif self.m_interp == "Hermite":
                xs = self.interp_hermite_func(x - x0)
                ys = self.interp_hermite_func(y - y0)
            else:  # Quintic
                xs = self.interp_quintic_func(x - x0)
                ys = self.interp_quintic_func(y - y0)

            xf0 = self.lerp(val_coord_2d(seed, x0, y0), val_coord_2d(seed, x1, y0), xs)
            xf1 = self.lerp(val_coord_2d(seed, x0, y1), val_coord_2d(seed, x1, y1), xs)

            return self.lerp(xf0, xf1, ys)

    def get_perlin_fractal(self, x, y, z=None):
        x *= self.m_frequency
        y *= self.m_frequency
        if z is not None:
            z *= self.m_frequency
            if self.m_fractalType == "FBM":
                return self.single_perlin_fractal_fbm(x, y, z)
            elif self.m_fractalType == "Billow":
                return self.single_perlin_fractal_billow(x, y, z)
            elif self.m_fractalType == "RigidMulti":
                return self.single_perlin_fractal_rigid_multi(x, y, z)
        else:
            if self.m_fractalType == "FBM":
                return self.single_perlin_fractal_fbm(x, y)
            elif self.m_fractalType == "Billow":
                return self.single_perlin_fractal_billow(x, y)
            elif self.m_fractalType == "RigidMulti":
                return self.single_perlin_fractal_rigid_multi(x, y)
        return 0

    def single_perlin_fractal_fbm(self, x, y, z=None):
        seed = self.m_seed
        sum_value = self.single_perlin(seed, x, y, z)
        amp = 1

        for _ in range(1, self.m_octaves):
            x *= self.m_lacunarity
            y *= self.m_lacunarity
            if z is not None:
                z *= self.m_lacunarity

            amp *= self.m_gain
            sum_value += self.single_perlin(seed + 1, x, y, z) * amp

        return sum_value * self.m_fractalBounding

    def single_perlin_fractal_billow(self, x, y, z=None):
        seed = self.m_seed
        sum_value = abs(self.single_perlin(seed, x, y, z)) * 2 - 1
        amp = 1

        for _ in range(1, self.m_octaves):
            x *= self.m_lacunarity
            y *= self.m_lacunarity
            if z is not None:
                z *= self.m_lacunarity

            amp *= self.m_gain
            sum_value += (abs(self.single_perlin(seed + 1, x, y, z)) * 2 - 1) * amp

        return sum_value * self.m_fractalBounding

    def single_perlin_fractal_rigid_multi(self, x, y, z=None):
        seed = self.m_seed
        sum_value = 1 - abs(self.single_perlin(seed, x, y, z))
        amp = 1

        for _ in range(1, self.m_octaves):
            x *= self.m_lacunarity
            y *= self.m_lacunarity
            if z is not None:
                z *= self.m_lacunarity

            amp *= self.m_gain
            sum_value -= (1 - abs(self.single_perlin(seed + 1, x, y, z))) * amp

        return sum_value

    def get_perlin(self, x, y, z=None):
        if z is not None:
            return self.single_perlin(self.m_seed, x * self.m_frequency, y * self.m_frequency, z * self.m_frequency)
        else:
            return self.single_perlin(self.m_seed, x * self.m_frequency, y * self.m_frequency)

    def single_perlin(self, seed, x, y, z=None):
        x0 = math.floor(x)
        y0 = math.floor(y)
        if z is not None:
            z0 = math.floor(z)
            x1 = x0 + 1
            y1 = y0 + 1
            z1 = z0 + 1

            if self.m_interp == "Linear":
                xs = x - x0
                ys = y - y0
                zs = z - z0
            elif self.m_interp == "Hermite":
                xs = self.interp_hermite_func(x - x0)
                ys = self.interp_hermite_func(y - y0)
                zs = self.interp_hermite_func(z - z0)
            else:  # Quintic
                xs = self.interp_quintic_func(x - x0)
                ys = self.interp_quintic_func(y - y0)
                zs = self.interp_quintic_func(z - z0)

            xd0 = x - x0
            yd0 = y - y0
            zd0 = z - z0
            xd1 = xd0 - 1
            yd1 = yd0 - 1
            zd1 = zd0 - 1

            xf00 = self.lerp(grad_coord_3d(seed, x0, y0, z0, xd0, yd0, zd0), grad_coord_3d(seed, x1, y0, z0, xd1, yd0, zd0), xs)
            xf10 = self.lerp(grad_coord_3d(seed, x0, y1, z0, xd0, yd1, zd0), grad_coord_3d(seed, x1, y1, z0, xd1, yd1, zd0), xs)
            xf01 = self.lerp(grad_coord_3d(seed, x0, y0, z1, xd0, yd0, zd1), grad_coord_3d(seed, x1, y0, z1, xd1, yd0, zd1), xs)
            xf11 = self.lerp(grad_coord_3d(seed, x0, y1, z1, xd0, yd1, zd1), grad_coord_3d(seed, x1, y1, z1, xd1, yd1, zd1), xs)

            yf0 = self.lerp(xf00, xf10, ys)
            yf1 = self.lerp(xf01, xf11, ys)

            return self.lerp(yf0, yf1, zs)
        else:
            x1 = x0 + 1
            y1 = y0 + 1

            if self.m_interp == "Linear":
                xs = x - x0
                ys = y - y0
            elif self.m_interp == "Hermite":
                xs = self.interp_hermite_func(x - x0)
                ys = self.interp_hermite_func(y - y0)
            else:  # Quintic
                xs = self.interp_quintic_func(x - x0)
                ys = self.interp_quintic_func(y - y0)

            xd0 = x - x0
            yd0 = y - y0
            xd1 = xd0 - 1
            yd1 = yd0 - 1

            xf0 = self.lerp(grad_coord_2d(seed, x0, y0, xd0, yd0), grad_coord_2d(seed, x1, y0, xd1, yd0), xs)
            xf1 = self.lerp(grad_coord_2d(seed, x0, y1, xd0, yd1), grad_coord_2d(seed, x1, y1, xd1, yd1), xs)

            return self.lerp(xf0, xf1, ys)

    @staticmethod
    def lerp(a, b, t):
        return a + t * (b - a)

    @staticmethod
    def interp_hermite_func(t):
        return t * t * (3 - 2 * t)

    @staticmethod
    def interp_quintic_func(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

import math

class Float2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Float3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

X_PRIME = 1619
Y_PRIME = 31337
Z_PRIME = 6971
W_PRIME = 1013
FN_DECIMAL = float

GRAD_3D = [
    Float3(1, 1, 0), Float3(-1, 1, 0), Float3(1, -1, 0), Float3(-1, -1, 0),
    Float3(1, 0, 1), Float3(-1, 0, 1), Float3(1, 0, -1), Float3(-1, 0, -1),
    Float3(0, 1, 1), Float3(0, -1, 1), Float3(0, 1, -1), Float3(0, -1, -1),
    Float3(1, 1, 0), Float3(0, -1, 1), Float3(-1, 1, 0), Float3(0, -1, -1),
]

F3 = 1.0 / 3.0
G3 = 1.0 / 6.0
G33 = G3 * 3 - 1

def grad_coord_3d(seed, x, y, z, xd, yd, zd):
    hash_val = seed
    hash_val ^= X_PRIME * x
    hash_val ^= Y_PRIME * y
    hash_val ^= Z_PRIME * z
    hash_val = hash_val * hash_val * hash_val * 60493
    hash_val = (hash_val >> 13) ^ hash_val
    g = GRAD_3D[hash_val & 15]
    return xd * g.x + yd * g.y + zd * g.z

def fast_floor(x):
    return int(x) if x > 0 else int(x) - 1

def lerp(a, b, t):
    return a + t * (b - a)

def interp_hermite_func(t):
    return t * t * (3 - 2 * t)

def interp_quintic_func(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

class FastNoise:
    def __init__(self, seed=1337):
        self.m_seed = seed
        self.m_frequency = 0.01
        self.m_interp = "Quintic"
        self.m_noiseType = "Simplex"
        self.m_octaves = 3
        self.m_lacunarity = 2.0
        self.m_gain = 0.5
        self.m_fractalType = "FBM"
        self.m_fractalBounding = 0
        self.calculate_fractal_bounding()

    def calculate_fractal_bounding(self):
        amp = self.m_gain
        amp_fractal = 1
        for _ in range(1, self.m_octaves):
            amp_fractal += amp
            amp *= self.m_gain
        self.m_fractalBounding = 1 / amp_fractal

    def get_simplex_fractal(self, x, y, z=None):
        x *= self.m_frequency
        y *= self.m_frequency
        if z is not None:
            z *= self.m_frequency
            if self.m_fractalType == "FBM":
                return self.single_simplex_fractal_fbm(x, y, z)
            elif self.m_fractalType == "Billow":
                return self.single_simplex_fractal_billow(x, y, z)
            elif self.m_fractalType == "RigidMulti":
                return self.single_simplex_fractal_rigid_multi(x, y, z)
        else:
            if self.m_fractalType == "FBM":
                return self.single_simplex_fractal_fbm(x, y)
            elif self.m_fractalType == "Billow":
                return self.single_simplex_fractal_billow(x, y)
            elif self.m_fractalType == "RigidMulti":
                return self.single_simplex_fractal_rigid_multi(x, y)
        return 0

    def single_simplex_fractal_fbm(self, x, y, z=None):
        seed = self.m_seed
        sum_value = self.single_simplex(seed, x, y, z)
        amp = 1

        for _ in range(1, self.m_octaves):
            x *= self.m_lacunarity
            y *= self.m_lacunarity
            if z is not None:
                z *= self.m_lacunarity

            amp *= self.m_gain
            sum_value += self.single_simplex(seed + 1, x, y, z) * amp

        return sum_value * self.m_fractalBounding

    def single_simplex_fractal_billow(self, x, y, z=None):
        seed = self.m_seed
        sum_value = abs(self.single_simplex(seed, x, y, z)) * 2 - 1
        amp = 1

        for _ in range(1, self.m_octaves):
            x *= self.m_lacunarity
            y *= self.m_lacunarity
            if z is not None:
                z *= self.m_lacunarity

            amp *= self.m_gain
            sum_value += (abs(self.single_simplex(seed + 1, x, y, z)) * 2 - 1) * amp

        return sum_value * self.m_fractalBounding

    def single_simplex_fractal_rigid_multi(self, x, y, z=None):
        seed = self.m_seed
        sum_value = 1 - abs(self.single_simplex(seed, x, y, z))
        amp = 1

        for _ in range(1, self.m_octaves):
            x *= self.m_lacunarity
            y *= self.m_lacunarity
            if z is not None:
                z *= self.m_lacunarity

            amp *= self.m_gain
            sum_value -= (1 - abs(self.single_simplex(seed + 1, x, y, z))) * amp

        return sum_value

    def get_simplex(self, x, y, z=None):
        if z is not None:
            return self.single_simplex(self.m_seed, x * self.m_frequency, y * self.m_frequency, z * self.m_frequency)
        else:
            return self.single_simplex(self.m_seed, x * self.m_frequency, y * self.m_frequency)

    def single_simplex(self, seed, x, y, z=None):
        if z is not None:
            t = (x + y + z) * F3
            i = fast_floor(x + t)
            j = fast_floor(y + t)
            k = fast_floor(z + t)

            t = (i + j + k) * G3
            x0 = x - (i - t)
            y0 = y - (j - t)
            z0 = z - (k - t)

            if x0 >= y0:
                if y0 >= z0:
                    i1, j1, k1 = 1, 0, 0
                    i2, j2, k2 = 1, 1, 0
                elif x0 >= z0:
                    i1, j1, k1 = 1, 0, 0
                    i2, j2, k2 = 1, 0, 1
                else:
                    i1, j1, k1 = 0, 0, 1
                    i2, j2, k2 = 1, 0, 1
            else:
                if y0 < z0:
                    i1, j1, k1 = 0, 0, 1
                    i2, j2, k2 = 0, 1, 1
                elif x0 < z0:
                    i1, j1, k1 = 0, 1, 0
                    i2, j2, k2 = 0, 1, 1
                else:
                    i1, j1, k1 = 0, 1, 0
                    i2, j2, k2 = 1, 1, 0

            x1 = x0 - i1 + G3
            y1 = y0 - j1 + G3
            z1 = z0 - k1 + G3
            x2 = x0 - i2 + F3
            y2 = y0 - j2 + F3
            z2 = z0 - k2 + F3
            x3 = x0 + G33
            y3 = y0 + G33
            z3 = z0 + G33

            n0, n1, n2, n3 = 0, 0, 0, 0

            t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0
            if t0 > 0:
                t0 *= t0
                n0 = t0 * t0 * grad_coord_3d(seed, i, j, k, x0, y0, z0)

            t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1
            if t1 > 0:
                t1 *= t1
                n1 = t1 * t1 * grad_coord_3d(seed, i + i1, j + j1, k + k1, x1, y1, z1)

            t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2
            if t2 > 0:
                t2 *= t2
                n2 = t2 * t2 * grad_coord_3d(seed, i + i2, j + j2, k + k2, x2, y2, z2)

            t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3
            if t3 > 0:
                t3 *= t3
                n3 = t3 * t3 * grad_coord_3d(seed, i + 1, j + 1, k + 1, x3, y3, z3)

            return 32 * (n0 + n1 + n2 + n3)
        else:
            # 2D simplex noise (to be implemented if required)
            pass

# Example usage:
noise = FastNoise()
print(noise.get_simplex(1.0, 2.0, 3.0))
print(noise.get_simplex_fractal(1.0, 2.0, 3.0))

import math

class Float2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Float3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class FastNoise:
    def __init__(self, seed=1337):
        self.m_seed = seed
        self.m_frequency = 0.01
        self.m_interp = "Quintic"
        self.m_noiseType = "Simplex"
        self.m_octaves = 3
        self.m_lacunarity = 2.0
        self.m_gain = 0.5
        self.m_fractalType = "FBM"
        self.calculate_fractal_bounding()

    def calculate_fractal_bounding(self):
        amp = self.m_gain
        amp_fractal = 1
        for _ in range(1, self.m_octaves):
            amp_fractal += amp
            amp *= self.m_gain
        self.m_fractalBounding = 1 / amp_fractal

    def get_simplex_fractal(self, x, y, z=None):
        x *= self.m_frequency
        y *= self.m_frequency
        if z is not None:
            z *= self.m_frequency
            if self.m_fractalType == "FBM":
                return self.single_simplex_fractal_fbm(x, y, z)
            elif self.m_fractalType == "Billow":
                return self.single_simplex_fractal_billow(x, y, z)
            elif self.m_fractalType == "RigidMulti":
                return self.single_simplex_fractal_rigid_multi(x, y, z)
        else:
            if self.m_fractalType == "FBM":
                return self.single_simplex_fractal_fbm(x, y)
            elif self.m_fractalType == "Billow":
                return self.single_simplex_fractal_billow(x, y)
            elif self.m_fractalType == "RigidMulti":
                return self.single_simplex_fractal_rigid_multi(x, y)
        return 0

    def single_simplex_fractal_fbm(self, x, y, z=None):
        seed = self.m_seed
        sum_value = self.single_simplex(seed, x, y, z)
        amp = 1

        for _ in range(1, self.m_octaves):
            x *= self.m_lacunarity
            y *= self.m_lacunarity
            if z is not None:
                z *= self.m_lacunarity

            amp *= self.m_gain
            sum_value += self.single_simplex(seed + 1, x, y, z) * amp

        return sum_value * self.m_fractalBounding

    def single_simplex_fractal_billow(self, x, y, z=None):
        seed = self.m_seed
        sum_value = abs(self.single_simplex(seed, x, y, z)) * 2 - 1
        amp = 1

        for _ in range(1, self.m_octaves):
            x *= self.m_lacunarity
            y *= self.m_lacunarity
            if z is not None:
                z *= self.m_lacunarity

            amp *= self.m_gain
            sum_value += (abs(self.single_simplex(seed + 1, x, y, z)) * 2 - 1) * amp

        return sum_value * self.m_fractalBounding

    def single_simplex_fractal_rigid_multi(self, x, y, z=None):
        seed = self.m_seed
        sum_value = 1 - abs(self.single_simplex(seed, x, y, z))
        amp = 1

        for _ in range(1, self.m_octaves):
            x *= self.m_lacunarity
            y *= self.m_lacunarity
            if z is not None:
                z *= self.m_lacunarity

            amp *= self.m_gain
            sum_value -= (1 - abs(self.single_simplex(seed + 1, x, y, z))) * amp

        return sum_value

    def get_simplex(self, x, y, z=None, w=None):
        if w is not None:
            return self.single_simplex(self.m_seed, x * self.m_frequency, y * self.m_frequency, z * self.m_frequency, w * self.m_frequency)
        elif z is not None:
            return self.single_simplex(self.m_seed, x * self.m_frequency, y * self.m_frequency, z * self.m_frequency)
        else:
            return self.single_simplex(self.m_seed, x * self.m_frequency, y * self.m_frequency)

    def single_simplex(self, seed, x, y, z=None, w=None):
        if w is not None:
            # 4D simplex noise
            return self.single_simplex_4d(seed, x, y, z, w)
        elif z is not None:
            # 3D simplex noise
            return self.single_simplex_3d(seed, x, y, z)
        else:
            # 2D simplex noise
            return self.single_simplex_2d(seed, x, y)

    def single_simplex_2d(self, seed, x, y):
        F2 = 1.0 / 2.0
        G2 = 1.0 / 4.0

        t = (x + y) * F2
        i = fast_floor(x + t)
        j = fast_floor(y + t)

        t = (i + j) * G2
        X0 = i - t
        Y0 = j - t

        x0 = x - X0
        y0 = y - Y0

        if x0 > y0:
            i1, j1 = 1, 0
        else:
            i1, j1 = 0, 1

        x1 = x0 - i1 + G2
        y1 = y0 - j1 + G2
        x2 = x0 - 1 + F2
        y2 = y0 - 1 + F2

        n0, n1, n2 = 0, 0, 0

        t0 = 0.5 - x0 * x0 - y0 * y0
        if t0 >= 0:
            t0 *= t0
            n0 = t0 * t0 * self.grad_coord_2d(seed, i, j, x0, y0)

        t1 = 0.5 - x1 * x1 - y1 * y1
        if t1 >= 0:
            t1 *= t1
            n1 = t1 * t1 * self.grad_coord_2d(seed, i + i1, j + j1, x1, y1)

        t2 = 0.5 - x2 * x2 - y2 * y2
        if t2 >= 0:
            t2 *= t2
            n2 = t2 * t2 * self.grad_coord_2d(seed, i + 1, j + 1, x2, y2)

        return 50 * (n0 + n1 + n2)

    def single_simplex_3d(self, seed, x, y, z):
        F3 = 1.0 / 3.0
        G3 = 1.0 / 6.0
        G33 = G3 * 3 - 1

        t = (x + y + z) * F3
        i = fast_floor(x + t)
        j = fast_floor(y + t)
        k = fast_floor(z + t)

        t = (i + j + k) * G3
        x0 = x - (i - t)
        y0 = y - (j - t)
        z0 = z - (k - t)

        if x0 >= y0:
            if y0 >= z0:
                i1, j1, k1 = 1, 0, 0
                i2, j2, k2 = 1, 1, 0
            elif x0 >= z0:
                i1, j1, k1 = 1, 0, 0
                i2, j2, k2 = 1, 0, 1
            else:
                i1, j1, k1 = 0, 0, 1
                i2, j2, k2 = 1, 0, 1
        else:
            if y0 < z0:
                i1, j1, k1 = 0, 0, 1
                i2, j2, k2 = 0, 1, 1
            elif x0 < z0:
                i1, j1, k1 = 0, 1, 0
                i2, j2, k2 = 0, 1, 1
            else:
                i1, j1, k1 = 0, 1, 0
                i2, j2, k2 = 1, 1, 0

        x1 = x0 - i1 + G3
        y1 = y0 - j1 + G3
        z1 = z0 - k1 + G3
        x2 = x0 - i2 + F3
        y2 = y0 - j2 + F3
        z2 = z0 - k2 + F3
        x3 = x0 + G33
        y3 = y0 + G33
        z3 = z0 + G33

        n0, n1, n2, n3 = 0, 0, 0, 0

        t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0
        if t0 > 0:
            t0 *= t0
            n0 = t0 * t0 * self.grad_coord_3d(seed, i, j, k, x0, y0, z0)

        t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1
        if t1 > 0:
            t1 *= t1
            n1 = t1 * t1 * self.grad_coord_3d(seed, i + i1, j + j1, k + k1, x1, y1, z1)

        t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2
        if t2 > 0:
            t2 *= t2
            n2 = t2 * t2 * self.grad_coord_3d(seed, i + i2, j + j2, k + k2, x2, y2, z2)

        t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3
        if t3 > 0:
            t3 *= t3
            n3 = t3 * t3 * self.grad_coord_3d(seed, i + 1, j + 1, k + 1, x3, y3, z3)

        return 32 * (n0 + n1 + n2 + n3)

    def single_simplex_4d(self, seed, x, y, z, w):
        F4 = (math.sqrt(5.0) - 1.0) / 4.0
        G4 = (5.0 - math.sqrt(5.0)) / 20.0

        t = (x + y + z + w) * F4
        i = fast_floor(x + t)
        j = fast_floor(y + t)
        k = fast_floor(z + t)
        l = fast_floor(w + t)
        t = (i + j + k + l) * G4
        X0 = i - t
        Y0 = j - t
        Z0 = k - t
        W0 = l - t
        x0 = x - X0
        y0 = y - Y0
        z0 = z - Z0
        w0 = w - W0

        c = [0, 0, 0, 0]
        if x0 > y0:
            c[0] += 1
        else:
            c[1] += 1
        if x0 > z0:
            c[0] += 1
        else:
            c[2] += 1
        if x0 > w0:
            c[0] += 1
        else:
            c[3] += 1
        if y0 > z0:
            c[1] += 1
        else:
            c[2] += 1
        if y0 > w0:
            c[1] += 1
        else:
            c[3] += 1
        if z0 > w0:
            c[2] += 1
        else:
            c[3] += 1

        i1, j1, k1, l1 = 0, 0, 0, 0
        i2, j2, k2, l2 = 0, 0, 0, 0
        i3, j3, k3, l3 = 0, 0, 0, 0

        if c[0] >= 3:
            i1 = 1
        elif c[0] >= 2:
            i2 = 1
        elif c[0] >= 1:
            i3 = 1
        if c[1] >= 3:
            j1 = 1
        elif c[1] >= 2:
            j2 = 1
        elif c[1] >= 1:
            j3 = 1
        if c[2] >= 3:
            k1 = 1
        elif c[2] >= 2:
            k2 = 1
        elif c[2] >= 1:
            k3 = 1
        if c[3] >= 3:
            l1 = 1
        elif c[3] >= 2:
            l2 = 1
        elif c[3] >= 1:
            l3 = 1

        x1 = x0 - i1 + G4
        y1 = y0 - j1 + G4
        z1 = z0 - k1 + G4
        w1 = w0 - l1 + G4
        x2 = x0 - i2 + 2.0 * G4
        y2 = y0 - j2 + 2.0 * G4
        z2 = z0 - k2 + 2.0 * G4
        w2 = w0 - l2 + 2.0 * G4
        x3 = x0 - i3 + 3.0 * G4
        y3 = y0 - j3 + 3.0 * G4
        z3 = z0 - k3 + 3.0 * G4
        w3 = w0 - l3 + 3.0 * G4
        x4 = x0 - 1.0 + 4.0 * G4
        y4 = y0 - 1.0 + 4.0 * G4
        z4 = z0 - 1.0 + 4.0 * G4
        w4 = w0 - 1.0 + 4.0 * G4

        n0, n1, n2, n3, n4 = 0, 0, 0, 0, 0

        t0 = 0.5 - x0 * x0 - y0 * y0 - z0 * z0 - w0 * w0
        if t0 >= 0:
            t0 *= t0
            n0 = t0 * t0 * self.grad_coord_4d(seed, i, j, k, l, x0, y0, z0, w0)

        t1 = 0.5 - x1 * x1 - y1 * y1 - z1 * z1 - w1 * w1
        if t1 >= 0:
            t1 *= t1
            n1 = t1 * t1 * self.grad_coord_4d(seed, i + i1, j + j1, k + k1, l + l1, x1, y1, z1, w1)

        t2 = 0.5 - x2 * x2 - y2 * y2 - z2 * z2 - w2 * w2
        if t2 >= 0:
            t2 *= t2
            n2 = t2 * t2 * self.grad_coord_4d(seed, i + i2, j + j2, k + k2, l + l2, x2, y2, z2, w2)

        t3 = 0.5 - x3 * x3 - y3 * y3 - z3 * z3 - w3 * w3
        if t3 >= 0:
            t3 *= t3
            n3 = t3 * t3 * self.grad_coord_4d(seed, i + i3, j + j3, k + k3, l + l3, x3, y3, z3, w3)

        t4 = 0.5 - x4 * x4 - y4 * y4 - z4 * z4 - w4 * w4
        if t4 >= 0:
            t4 *= t4
            n4 = t4 * t4 * self.grad_coord_4d(seed, i + 1, j + 1, k + 1, l + 1, x4, y4, z4, w4)

        return 27 * (n0 + n1 + n2 + n3 + n4)

    @staticmethod
    def fast_floor(f):
        return int(f) if f >= 0 else int(f) - 1

    @staticmethod
    def interp_hermite_func(t):
        return t * t * (3 - 2 * t)

    @staticmethod
    def interp_quintic_func(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    @staticmethod
    def lerp(a, b, t):
        return a + t * (b - a)

    @staticmethod
    def grad_coord_2d(seed, x, y, xd, yd):
        hash = seed
        hash ^= 1619 * x
        hash ^= 31337 * y

        hash = hash * hash * hash * 60493
        hash = (hash >> 13) ^ hash

        g = [Float2(-1, -1), Float2(1, -1), Float2(-1, 1), Float2(1, 1),
             Float2(0, -1), Float2(-1, 0), Float2(0, 1), Float2(1, 0)][hash & 7]

        return xd * g.x + yd * g.y

    @staticmethod
    def grad_coord_3d(seed, x, y, z, xd, yd, zd):
        hash = seed
        hash ^= 1619 * x
        hash ^= 31337 * y
        hash ^= 6971 * z

        hash = hash * hash * hash * 60493
        hash = (hash >> 13) ^ hash

        g = [Float3(1, 1, 0), Float3(-1, 1, 0), Float3(1, -1, 0), Float3(-1, -1, 0),
             Float3(1, 0, 1), Float3(-1, 0, 1), Float3(1, 0, -1), Float3(-1, 0, -1),
             Float3(0, 1, 1), Float3(0, -1, 1), Float3(0, 1, -1), Float3(0, -1, -1),
             Float3(1, 1, 0), Float3(0, -1, 1), Float3(-1, 1, 0), Float3(0, -1, -1)][hash & 15]

        return xd * g.x + yd * g.y + zd * g.z

    @staticmethod
    def grad_coord_4d(seed, x, y, z, w, xd, yd, zd, wd):
        hash = seed
        hash ^= 1619 * x
        hash ^= 31337 * y
        hash ^= 6971 * z
        hash ^= 1013 * w

        hash = hash * hash * hash * 60493
        hash = (hash >> 13) ^ hash

        hash &= 31
        a, b, c = yd, zd, wd
        if hash >> 3 == 1:
            a, b, c = wd, xd, yd
        elif hash >> 3 == 2:
            a, b, c = zd, wd, xd
        elif hash >> 3 == 3:
            a, b, c = yd, zd, wd

        return ((hash & 4) == 0) * -a + ((hash & 2) == 0) * -b + ((hash & 1) == 0) * -c

import cupy as cp
import numpy as np

class FastNoise:
    def __init__(self, seed, frequency, octaves, lacunarity, gain, fractal_type, interp):
        self.seed = seed
        self.frequency = frequency
        self.octaves = octaves
        self.lacunarity = lacunarity
        self.gain = gain
        self.fractal_type = fractal_type
        self.interp = interp
        self.fractal_bounding = self.calculate_fractal_bounding()

    @staticmethod
    def fast_floor(x):
        return cp.floor(x).astype(int)

    @staticmethod
    def lerp(a, b, t):
        return a + t * (b - a)

    @staticmethod
    def interp_hermite_func(t):
        return t * t * (3 - 2 * t)

    @staticmethod
    def interp_quintic_func(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def calculate_fractal_bounding(self):
        amp = self.gain
        amp_fractal = 1.0
        for _ in range(self.octaves - 1):
            amp_fractal += amp
            amp *= self.gain
        return 1.0 / amp_fractal

    def grad_coord_3d(self, seed, x, y, z, xd, yd, zd):
        hash = seed
        hash ^= 1619 * x
        hash ^= 31337 * y
        hash ^= 6971 * z
        hash = hash * hash * hash * 60493
        hash = (hash >> 13) ^ hash
        g = cp.array([
            (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
            (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
            (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
            (1, 1, 0), (0, -1, 1), (-1, 1, 0), (0, -1, -1)
        ])
        g = g[hash & 15]
        return xd * g[0] + yd * g[1] + zd * g[2]

    def single_cubic(self, seed, x, y, z):
        x1 = self.fast_floor(x)
        y1 = self.fast_floor(y)
        z1 = self.fast_floor(z)

        x0 = x1 - 1
        y0 = y1 - 1
        z0 = z1 - 1
        x2 = x1 + 1
        y2 = y1 + 1
        z2 = z1 + 1
        x3 = x1 + 2
        y3 = y1 + 2
        z3 = z1 + 2

        xs = self.interp_quintic_func(x - x1)
        ys = self.interp_quintic_func(y - y1)
        zs = self.interp_quintic_func(z - z1)

        def val_coord_3d(seed, x, y, z):
            n = seed
            n ^= 1619 * x
            n ^= 31337 * y
            n ^= 6971 * z
            n = n * n * n * 60493
            return (n >> 13) ^ n

        def val_lerp(seed, x, y, z, x0, x1, t):
            return self.lerp(val_coord_3d(seed, x, y, z), val_coord_3d(seed, x + 1, y, z), t)

        def cubic_lerp(p):
            return (
                p[0] * 0.5 + p[1] * 1.5 - p[2] * 1.5 + p[3] * 0.5
            )

        def sample_lerp(seed, x, y, z):
            return cubic_lerp([
                cubic_lerp([val_lerp(seed, x, y, z, x0, x3, xs),
                            val_lerp(seed, x, y, z, x1, x2, xs)]),
                cubic_lerp([val_lerp(seed, x, y, z, x0, x3, xs),
                            val_lerp(seed, x, y, z, x1, x2, xs)])
            ])

        return sample_lerp(seed, x, y, z)

    def get_cubic(self, x, y, z):
        return self.single_cubic(self.seed, x * self.frequency, y * self.frequency, z * self.frequency)

    def get_cubic_fractal(self, x, y, z):
        x *= self.frequency
        y *= self.frequency
        z *= self.frequency

        if self.fractal_type == 'FBM':
            return self.single_cubic_fractal_fbm(x, y, z)
        elif self.fractal_type == 'Billow':
            return self.single_cubic_fractal_billow(x, y, z)
        elif self.fractal_type == 'RigidMulti':
            return self.single_cubic_fractal_rigid_multi(x, y, z)
        else:
            return 0

    def single_cubic_fractal_fbm(self, x, y, z):
        seed = self.seed
        sum = self.single_cubic(seed, x, y, z)
        amp = 1
        for i in range(1, self.octaves):
            x *= self.lacunarity
            y *= self.lacunarity
            z *= self.lacunarity
            amp *= self.gain
            sum += self.single_cubic(seed + i, x, y, z) * amp
        return sum * self.fractal_bounding

    def single_cubic_fractal_billow(self, x, y, z):
        seed = self.seed
        sum = abs(self.single_cubic(seed, x, y, z)) * 2 - 1
        amp = 1
        for i in range(1, self.octaves):
            x *= self.lacunarity
            y *= self.lacunarity
            z *= self.lacunarity
            amp *= self.gain
            sum += (abs(self.single_cubic(seed + i, x, y, z)) * 2 - 1) * amp
        return sum * self.fractal_bounding

    def single_cubic_fractal_rigid_multi(self, x, y, z):
        seed = self.seed
        sum = 1 - abs(self.single_cubic(seed, x, y, z))
        amp = 1
        for i in range(1, self.octaves):
            x *= self.lacunarity
            y *= self.lacunarity
            z *= self.lacunarity
            amp *= self.gain
            sum -= (1 - abs(self.single_cubic(seed + i, x, y, z))) * amp
        return sum

import cupy as cp
import numpy as np

class FastNoise:
    def __init__(self, seed, frequency, octaves, lacunarity, gain, fractal_type, interp):
        self.seed = seed
        self.frequency = frequency
        self.octaves = octaves
        self.lacunarity = lacunarity
        self.gain = gain
        self.fractal_type = fractal_type
        self.interp = interp
        self.fractal_bounding = self.calculate_fractal_bounding()

    @staticmethod
    def fast_floor(x):
        return cp.floor(x).astype(int)

    @staticmethod
    def lerp(a, b, t):
        return a + t * (b - a)

    @staticmethod
    def interp_hermite_func(t):
        return t * t * (3 - 2 * t)

    @staticmethod
    def interp_quintic_func(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def calculate_fractal_bounding(self):
        amp = self.gain
        amp_fractal = 1.0
        for _ in range(self.octaves - 1):
            amp_fractal += amp
            amp *= self.gain
        return 1.0 / amp_fractal

    def val_coord_3d(self, seed, x, y, z):
        n = seed
        n ^= 1619 * x
        n ^= 31337 * y
        n ^= 6971 * z
        n = n * n * n * 60493
        n = (n >> 13) ^ n
        return (n / 2147483648.0)

    def cubic_lerp(self, a, b, c, d, t):
        p = (d - c) - (a - b)
        return t * t * t * p + t * t * ((a - b) - p) + t * (c - a) + b

    def single_cubic(self, seed, x, y, z):
        x1 = self.fast_floor(x)
        y1 = self.fast_floor(y)
        z1 = self.fast_floor(z)

        x0 = x1 - 1
        y0 = y1 - 1
        z0 = z1 - 1
        x2 = x1 + 1
        y2 = y1 + 1
        z2 = z1 + 1
        x3 = x1 + 2
        y3 = y1 + 2
        z3 = z1 + 2

        xs = self.interp_quintic_func(x - x1)
        ys = self.interp_quintic_func(y - y1)
        zs = self.interp_quintic_func(z - z1)

        return self.cubic_lerp(
            self.cubic_lerp(
                self.cubic_lerp(self.val_coord_3d(seed, x0, y0, z0), self.val_coord_3d(seed, x1, y0, z0),
                                self.val_coord_3d(seed, x2, y0, z0), self.val_coord_3d(seed, x3, y0, z0), xs),
                self.cubic_lerp(self.val_coord_3d(seed, x0, y1, z0), self.val_coord_3d(seed, x1, y1, z0),
                                self.val_coord_3d(seed, x2, y1, z0), self.val_coord_3d(seed, x3, y1, z0), xs),
                self.cubic_lerp(self.val_coord_3d(seed, x0, y2, z0), self.val_coord_3d(seed, x1, y2, z0),
                                self.val_coord_3d(seed, x2, y2, z0), self.val_coord_3d(seed, x3, y2, z0), xs),
                self.cubic_lerp(self.val_coord_3d(seed, x0, y3, z0), self.val_coord_3d(seed, x1, y3, z0),
                                self.val_coord_3d(seed, x2, y3, z0), self.val_coord_3d(seed, x3, y3, z0), xs),
                ys),
            self.cubic_lerp(
                self.cubic_lerp(self.val_coord_3d(seed, x0, y0, z1), self.val_coord_3d(seed, x1, y0, z1),
                                self.val_coord_3d(seed, x2, y0, z1), self.val_coord_3d(seed, x3, y0, z1), xs),
                self.cubic_lerp(self.val_coord_3d(seed, x0, y1, z1), self.val_coord_3d(seed, x1, y1, z1),
                                self.val_coord_3d(seed, x2, y1, z1), self.val_coord_3d(seed, x3, y1, z1), xs),
                self.cubic_lerp(self.val_coord_3d(seed, x0, y2, z1), self.val_coord_3d(seed, x1, y2, z1),
                                self.val_coord_3d(seed, x2, y2, z1), self.val_coord_3d(seed, x3, y2, z1), xs),
                self.cubic_lerp(self.val_coord_3d(seed, x0, y3, z1), self.val_coord_3d(seed, x1, y3, z1),
                                self.val_coord_3d(seed, x2, y3, z1), self.val_coord_3d(seed, x3, y3, z1), xs),
                ys),
            self.cubic_lerp(
                self.cubic_lerp(self.val_coord_3d(seed, x0, y0, z2), self.val_coord_3d(seed, x1, y0, z2),
                                self.val_coord_3d(seed, x2, y0, z2), self.val_coord_3d(seed, x3, y0, z2), xs),
                self.cubic_lerp(self.val_coord_3d(seed, x0, y1, z2), self.val_coord_3d(seed, x1, y1, z2),
                                self.val_coord_3d(seed, x2, y1, z2), self.val_coord_3d(seed, x3, y1, z2), xs),
                self.cubic_lerp(self.val_coord_3d(seed, x0, y2, z2), self.val_coord_3d(seed, x1, y2, z2),
                                self.val_coord_3d(seed, x2, y2, z2), self.val_coord_3d(seed, x3, y2, z2), xs),
                self.cubic_lerp(self.val_coord_3d(seed, x0, y3, z2), self.val_coord_3d(seed, x1, y3, z2),
                                self.val_coord_3d(seed, x2, y3, z2), self.val_coord_3d(seed, x3, y3, z2), xs),
                ys),
            self.cubic_lerp(
                self.cubic_lerp(self.val_coord_3d(seed, x0, y0, z3), self.val_coord_3d(seed, x1, y0, z3),
                                self.val_coord_3d(seed, x2, y0, z3), self.val_coord_3d(seed, x3, y0, z3), xs),
                self.cubic_lerp(self.val_coord_3d(seed, x0, y1, z3), self.val_coord_3d(seed, x1, y1, z3),
                                self.val_coord_3d(seed, x2, y1, z3), self.val_coord_3d(seed, x3, y1, z3), xs),
                self.cubic_lerp(self.val_coord_3d(seed, x0, y2, z3), self.val_coord_3d(seed, x1, y2, z3),
                                self.val_coord_3d(seed, x2, y2, z3), self.val_coord_3d(seed, x3, y2, z3), xs),
                self.cubic_lerp(self.val_coord_3d(seed, x0, y3, z3), self.val_coord_3d(seed, x1, y3, z3),
                                self.val_coord_3d(seed, x2, y3, z3), self.val_coord_3d(seed, x3, y3, z3), xs),
                ys),
            zs) * (1 / (1.5 * 1.5 * 1.5))

    def get_cubic_fractal(self, x, y):
        x *= self.frequency
        y *= self.frequency

        if self.fractal_type == 'FBM':
            return self.single_cubic_fractal_fbm(x, y)
        elif self.fractal_type == 'Billow':
            return self.single_cubic_fractal_billow(x, y)
        elif self.fractal_type == 'RigidMulti':
            return self.single_cubic_fractal_rigid_multi(x, y)
        else:
            return 0

    def single_cubic_fractal_fbm(self, x, y):
        seed = self.seed
        sum = self.single_cubic(seed, x, y, 0)
        amp = 1
        for i in range(1, self.octaves):
            x *= self.lacunarity
            y *= self.lacunarity
            amp *= self.gain
            sum += self.single_cubic(seed + i, x, y, 0) * amp
        return sum * self.fractal_bounding

    def single_cubic_fractal_billow(self, x, y):
        seed = self.seed
        sum = abs(self.single_cubic(seed, x, y, 0)) * 2 - 1
        amp = 1
        for i in range(1, self.octaves):
            x *= self.lacunarity
            y *= self.lacunarity
            amp *= self.gain
            sum += (abs(self.single_cubic(seed + i, x, y, 0)) * 2 - 1) * amp
        return sum * self.fractal_bounding

    def single_cubic_fractal_rigid_multi(self, x, y):
        seed = self.seed
        sum = 1 - abs(self.single_cubic(seed, x, y, 0))
        amp = 1
        for i in range(1, self.octaves):
            x *= self.lacunarity
            y *= self.lacunarity
            amp *= self.gain
            sum -= (1 - abs(self.single_cubic(seed + i, x, y, 0))) * amp
        return sum

import cupy as cp
import numpy as np

class FastNoise:
    def __init__(self, seed, frequency, octaves, lacunarity, gain, fractal_type, interp):
        self.seed = seed
        self.frequency = frequency
        self.octaves = octaves
        self.lacunarity = lacunarity
        self.gain = gain
        self.fractal_type = fractal_type
        self.interp = interp
        self.fractal_bounding = self.calculate_fractal_bounding()

    @staticmethod
    def fast_floor(x):
        return cp.floor(x).astype(int)

    @staticmethod
    def lerp(a, b, t):
        return a + t * (b - a)

    @staticmethod
    def interp_hermite_func(t):
        return t * t * (3 - 2 * t)

    @staticmethod
    def interp_quintic_func(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def calculate_fractal_bounding(self):
        amp = self.gain
        amp_fractal = 1.0
        for _ in range(self.octaves - 1):
            amp_fractal += amp
            amp *= self.gain
        return 1.0 / amp_fractal

    def val_coord_3d(self, seed, x, y, z):
        n = seed
        n ^= 1619 * x
        n ^= 31337 * y
        n ^= 6971 * z
        n = n * n * n * 60493
        n = (n >> 13) ^ n
        return (n / 2147483648.0)

    def cubic_lerp(self, a, b, c, d, t):
        p = (d - c) - (a - b)
        return t * t * t * p + t * t * ((a - b) - p) + t * (c - a) + b

    def single_cubic(self, seed, x, y, z):
        x1 = self.fast_floor(x)
        y1 = self.fast_floor(y)
        z1 = self.fast_floor(z)

        x0 = x1 - 1
        y0 = y1 - 1
        z0 = z1 - 1
        x2 = x1 + 1
        y2 = y1 + 1
        z2 = z1 + 1
        x3 = x1 + 2
        y3 = y1 + 2
        z3 = z1 + 2

        xs = self.interp_quintic_func(x - x1)
        ys = self.interp_quintic_func(y - y1)
        zs = self.interp_quintic_func(z - z1)

        return self.cubic_lerp(
            self.cubic_lerp(
                self.cubic_lerp(self.val_coord_3d(seed, x0, y0, z0), self.val_coord_3d(seed, x1, y0, z0),
                                self.val_coord_3d(seed, x2, y0, z0), self.val_coord_3d(seed, x3, y0, z0), xs),
                self.cubic_lerp(self.val_coord_3d(seed, x0, y1, z0), self.val_coord_3d(seed, x1, y1, z0),
                                self.val_coord_3d(seed, x2, y1, z0), self.val_coord_3d(seed, x3, y1, z0), xs),
                self.cubic_lerp(self.val_coord_3d(seed, x0, y2, z0), self.val_coord_3d(seed, x1, y2, z0),
                                self.val_coord_3d(seed, x2, y2, z0), self.val_coord_3d(seed, x3, y2, z0), xs),
                self.cubic_lerp(self.val_coord_3d(seed, x0, y3, z0), self.val_coord_3d(seed, x1, y3, z0),
                                self.val_coord_3d(seed, x2, y3, z0), self.val_coord_3d(seed, x3, y3, z0), xs),
                ys),
            self.cubic_lerp(
                self.cubic_lerp(self.val_coord_3d(seed, x0, y0, z1), self.val_coord_3d(seed, x1, y0, z1),
                                self.val_coord_3d(seed, x2, y0, z1), self.val_coord_3d(seed, x3, y0, z1), xs),
                self.cubic_lerp(self.val_coord_3d(seed, x0, y1, z1), self.val_coord_3d(seed, x1, y1, z1),
                                self.val_coord_3d(seed, x2, y1, z1), self.val_coord_3d(seed, x3, y1, z1), xs),
                self.cubic_lerp(self.val_coord_3d(seed, x0, y2, z1), self.val_coord_3d(seed, x1, y2, z1),
                                self.val_coord_3d(seed, x2, y2, z1), self.val_coord_3d(seed, x3, y2, z1), xs),
                self.cubic_lerp(self.val_coord_3d(seed, x0, y3, z1), self.val_coord_3d(seed, x1, y3, z1),
                                self.val_coord_3d(seed, x2, y3, z1), self.val_coord_3d(seed, x3, y3, z1), xs),
                ys),
            self.cubic_lerp(
                self.cubic_lerp(self.val_coord_3d(seed, x0, y0, z2), self.val_coord_3d(seed, x1, y0, z2),
                                self.val_coord_3d(seed, x2, y0, z2), self.val_coord_3d(seed, x3, y0, z2), xs),
                self.cubic_lerp(self.val_coord_3d(seed, x0, y1, z2), self.val_coord_3d(seed, x1, y1, z2),
                                self.val_coord_3d(seed, x2, y1, z2), self.val_coord_3d(seed, x3, y1, z2), xs),
                self.cubic_lerp(self.val_coord_3d(seed, x0, y2, z2), self.val_coord_3d(seed, x1, y2, z2),
                                self.val_coord_3d(seed, x2, y2, z2), self.val_coord_3d(seed, x3, y2, z2), xs),
                self.cubic_lerp(self.val_coord_3d(seed, x0, y3, z2), self.val_coord_3d(seed, x1, y3, z2),
                                self.val_coord_3d(seed, x2, y3, z2), self.val_coord_3d(seed, x3, y3, z2), xs),
                ys),
            self.cubic_lerp(
                self.cubic_lerp(self.val_coord_3d(seed, x0, y0, z3), self.val_coord_3d(seed, x1, y0, z3),
                                self.val_coord_3d(seed, x2, y0, z3), self.val_coord_3d(seed, x3, y0, z3), xs),
                self.cubic_lerp(self.val_coord_3d(seed, x0, y1, z3), self.val_coord_3d(seed, x1, y1, z3),
                                self.val_coord_3d(seed, x2, y1, z3), self.val_coord_3d(seed, x3, y1, z3), xs),
                self.cubic_lerp(self.val_coord_3d(seed, x0, y2, z3), self.val_coord_3d(seed, x1, y2, z3),
                                self.val_coord_3d(seed, x2, y2, z3), self.val_coord_3d(seed, x3, y2, z3), xs),
                self.cubic_lerp(self.val_coord_3d(seed, x0, y3, z3), self.val_coord_3d(seed, x1, y3, z3),
                                self.val_coord_3d(seed, x2, y3, z3), self.val_coord_3d(seed, x3, y3, z3), xs),
                ys),
            zs) * (1 / (1.5 * 1.5 * 1.5))

    def get_cubic_fractal(self, x, y):
        x *= self.frequency
        y *= self.frequency

        if self.fractal_type == 'FBM':
            return self.single_cubic_fractal_fbm(x, y)
        elif self.fractal_type == 'Billow':
            return self.single_cubic_fractal_billow(x, y)
        elif self.fractal_type == 'RigidMulti':
            return self.single_cubic_fractal_rigid_multi(x, y)
        else:
            return 0

    def single_cubic_fractal_fbm(self, x, y):
        seed = self.seed
        sum = self.single_cubic(seed, x, y, 0)
        amp = 1
        for i in range(1, self.octaves):
            x *= self.lacunarity
            y *= self.lacunarity
            amp *= self.gain
            sum += self.single_cubic(seed + i, x, y, 0) * amp
        return sum * self.fractal_bounding

    def single_cubic_fractal_billow(self, x, y):
        seed = self.seed
        sum = abs(self.single_cubic(seed, x, y, 0)) * 2 - 1
        amp = 1
        for i in range(1, self.octaves):
            x *= self.lacunarity
            y *= self.lacunarity
            amp *= self.gain
            sum += (abs(self.single_cubic(seed + i, x, y, 0)) * 2 - 1) * amp
        return sum * self.fractal_bounding

    def single_cubic_fractal_rigid_multi(self, x, y):
        seed = self.seed
        sum = 1 - abs(self.single_cubic(seed, x, y, 0))
        amp = 1
        for i in range(1, self.octaves):
            x *= self.lacunarity
            y *= self.lacunarity
            amp *= self.gain
            sum -= (1 - abs(self.single_cubic(seed + i, x, y, 0))) * amp
        return sum


from dataclasses import dataclass, field
from typing import List
import cupy as cp

@dataclass
class DensityProfileLayer:
    width: float
    exp_term: float
    exp_scale: float
    linear_term: float
    constant_term: float

@dataclass
class Model:
    wavelengths: List[float]
    solar_irradiance: List[float]
    sun_angular_radius: float
    bottom_radius: float
    top_radius: float
    rayleigh_density: DensityProfileLayer
    rayleigh_scattering: List[float]
    mie_density: DensityProfileLayer
    mie_scattering: List[float]
    mie_extinction: List[float]
    mie_phase_function_g: float
    absorption_density: List[DensityProfileLayer]
    absorption_extinction: List[float]
    ground_albedo: List[float]
    max_sun_zenith_angle: float
    length_unit_in_meters: float
    combine_scattering_textures: bool
    use_luminance: str
    half_precision: bool

    k_lambda_r: float = 680.0
    k_lambda_g: float = 550.0
    k_lambda_b: float = 440.0
    k_lambda_min: int = 360
    k_lambda_max: int = 830

    def __post_init__(self):
        self.num_precomputed_wavelengths = 15 if self.use_luminance == 'PRECOMPUTED' else 3

    @property
    def transmittance_texture(self):
        return cp.zeros((256, 256), dtype=cp.float16 if self.half_precision else cp.float32)

    @property
    def scattering_texture(self):
        return cp.zeros((256, 128, 32), dtype=cp.float16 if self.half_precision else cp.float32)

    @property
    def irradiance_texture(self):
        return cp.zeros((64, 16), dtype=cp.float16 if self.half_precision else cp.float32)

    @property
    def optional_single_mie_scattering_texture(self):
        return cp.zeros((256, 128, 32), dtype=cp.float16 if self.half_precision else cp.float32) if not self.combine_scattering_textures else None

    # Additional methods for calculations can be added here

# Example usage
rayleigh_density = DensityProfileLayer(width=1.0, exp_term=0.0, exp_scale=1.0, linear_term=0.0, constant_term=1.0)
mie_density = DensityProfileLayer(width=1.0, exp_term=0.0, exp_scale=1.0, linear_term=0.0, constant_term=1.0)
absorption_density = [DensityProfileLayer(width=1.0, exp_term=0.0, exp_scale=1.0, linear_term=0.0, constant_term=1.0)]

model = Model(
    wavelengths=[680.0, 550.0, 440.0],
    solar_irradiance=[1.5, 1.5, 1.5],
    sun_angular_radius=0.00935,
    bottom_radius=6360000.0,
    top_radius=6460000.0,
    rayleigh_density=rayleigh_density,
    rayleigh_scattering=[0.002, 0.003, 0.004],
    mie_density=mie_density,
    mie_scattering=[0.004, 0.005, 0.006],
    mie_extinction=[0.008, 0.009, 0.010],
    mie_phase_function_g=0.76,
    absorption_density=absorption_density,
    absorption_extinction=[0.002, 0.003, 0.004],
    ground_albedo=[0.1, 0.1, 0.1],
    max_sun_zenith_angle=102.0,
    length_unit_in_meters=1000.0,
    combine_scattering_textures=True,
    use_luminance='PRECOMPUTED',
    half_precision=False
)

print(model.transmittance_texture.shape)
print(model.scattering_texture.shape)
print(model.irradiance_texture.shape)
if model.optional_single_mie_scattering_texture is not None:
    print(model.optional_single_mie_scattering_texture.shape)

import numpy as np
import cupy as cp

class FoliageExperimental:
    def __init__(self, planet):
        self.planet = planet
        self.max_points_per_frame = 1000
        self.size = 1500.0
        self.use_raycast = False
        self.radius_sqr = 2 * self.size * self.size

        self.grass = None
        self.detail_meshes = None
        self.detail_prefabs = None
        self.point_cloud = None
        self.matrices = None
        self.transforms = None

        self.last_position = cp.zeros(3)
        self.last_planet_position = cp.zeros(3)
        self.last_planet_rotation = cp.zeros(4)
        self.avg_pos = cp.zeros(3)

        self.generate_grass = False
        self.generate_meshes = False
        self.generate_prefabs = False

        self.initialized = False
        self.quad_size = 0
        self.quad_size_1 = 0

    class SurfacePoints:
        def __init__(self, number_of_points):
            self.number_of_points = number_of_points
            self.currently_used_points = 0
            self.points = cp.zeros((number_of_points, 3))
            self.normals = cp.zeros((number_of_points, 3))

    def awake(self):
        self.radius_sqr = 2 * self.size * self.size

    def initialize(self):
        self.generate_grass = self.planet.exp_grass
        self.generate_meshes = self.planet.exp_meshes
        self.generate_prefabs = self.planet.exp_prefabs

        self.quad_size = self.planet.quad_size
        self.quad_size_1 = self.quad_size - 1

        if self.generate_grass and self.planet.grass_per_quad > 0:
            self.grass = self.SurfacePoints(self.planet.grass_per_quad)
            self.point_cloud = cp.zeros((self.planet.grass_per_quad, 3))
        else:
            self.generate_grass = False

        number_detail_meshes = sum(mesh.number for mesh in self.planet.detail_meshes)
        if self.generate_meshes and number_detail_meshes > 0:
            self.detail_meshes = self.SurfacePoints(number_detail_meshes)
            self.matrices = cp.zeros((number_detail_meshes, 4, 4))
        else:
            self.generate_meshes = False

        number_detail_prefabs = sum(prefab.number for prefab in self.planet.detail_prefabs)
        if self.generate_prefabs and number_detail_prefabs > 0:
            self.detail_prefabs = self.SurfacePoints(number_detail_prefabs)
            self.transforms = [None] * number_detail_prefabs
        else:
            self.generate_prefabs = False

        offset = 0
        for prefab in self.planet.detail_prefabs:
            for j in range(prefab.number):
                self.transforms[j + offset] = prefab.prefab
            offset += prefab.number

        self.initialized = True

    def reset(self):
        self.grass = None
        self.detail_meshes = None
        self.detail_prefabs = None
        self.matrices = None
        self.transforms = None
        self.initialized = False

    def update_meshes(self):
        offset = 0
        rotation = self.calculate_rotation(self.detail_meshes.points[0])
        for mesh in self.planet.detail_meshes:
            for j in range(mesh.number):
                if not cp.allclose(self.detail_meshes.points[j + offset], 0):
                    self.matrices[j + offset] = self.calculate_matrix(self.detail_meshes.points[j + offset], mesh.mesh_offset_up, self.detail_meshes.normals[j + offset], rotation, mesh.mesh_scale)
            offset += mesh.number

    def render_meshes(self):
        offset = 0
        for mesh in self.planet.detail_meshes:
            if mesh.use_gpu_instancing:
                for j in range(mesh.number):
                    if not cp.allclose(self.detail_meshes.points[j + offset], 0):
                        # Replace with GPU draw call
                        pass
            else:
                to_render = [self.matrices[j + offset] for j in range(mesh.number) if not cp.allclose(self.detail_meshes.points[j + offset], 0)]
                # Replace with instanced draw call
            offset += mesh.number

    def update_transforms(self):
        offset = 0
        rotation = self.calculate_rotation(self.detail_prefabs.points[0])
        for prefab in self.planet.detail_prefabs:
            for j in range(prefab.number):
                if not cp.allclose(self.detail_prefabs.points[j + offset], 0):
                    self.transforms[j + offset].position = self.detail_prefabs.points[j + offset] + prefab.mesh_offset_up * self.detail_prefabs.normals[j + offset]
                    self.transforms[j + offset].rotation = rotation
                    self.transforms[j + offset].set_active(True)
                else:
                    self.transforms[j + offset].set_active(False)
            offset += prefab.number

    def update(self):
        tr_position = cp.asarray(self.get_position())
        down = self.calculate_down_direction(tr_position)
        hit = self.check_raycast(tr_position, down)
        if not hit:
            if self.initialized:
                self.reset()
            return
        if hit.tag == "Quad":
            biome = self.planet.find_quad(int(hit.name[5:])).biome
            if not self.is_biome_included(biome):
                return
            if not self.initialized:
                self.initialize()

        self.update_position_and_rotation(tr_position, down)

        if self.is_position_significantly_different(tr_position):
            self.last_position = tr_position
            self.reset_unused_points(tr_position)
            self.avg_pos = self.calculate_avg_pos(tr_position)

        if self.should_generate_foliage():
            if self.use_raycast:
                self.generate_foliage_raycast(self.avg_pos)
            else:
                self.generate_foliage(self.avg_pos)
            self.update_point_cloud()
            self.update_meshes()
            self.update_transforms()
        self.render_point_cloud()
        self.render_meshes()

    def calculate_rotation(self, point):
        return cp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    def calculate_matrix(self, point, offset, normal, rotation, scale):
        return cp.eye(4)

    def get_position(self):
        return cp.array([0, 0, 0])

    def calculate_down_direction(self, position):
        return cp.array([0, -1, 0])

    def check_raycast(self, position, direction):
        return None

    def is_biome_included(self, biome):
        return True

    def update_position_and_rotation(self, position, direction):
        delta_pos = self.planet.position - self.last_planet_position
        self.last_position -= delta_pos
        self.update_points(self.grass, delta_pos)
        self.update_points(self.detail_meshes, delta_pos)
        self.update_points(self.detail_prefabs, delta_pos)
        self.last_planet_position = self.planet.position

        rotate = self.calculate_rotation_difference()
        self.rotate_points(self.grass, rotate)
        self.rotate_points(self.detail_meshes, rotate)
        self.rotate_points(self.detail_prefabs, rotate)
        self.last_planet_rotation = self.planet.rotation

    def update_points(self, points, delta_pos):
        if points:
            for i in range(points.points.shape[0]):
                if not cp.allclose(points.points[i], 0):
                    points.points[i] += delta_pos

    def calculate_rotation_difference(self):
        return cp.array([1, 0, 0, 0])

    def rotate_points(self, points, rotate):
        if points:
            for i in range(points.points.shape[0]):
                if not cp.allclose(points.points[i], 0):
                    points.points[i] = self.rotate_point(points.points[i], rotate)

    def rotate_point(self, point, rotate):
        return point

    def is_position_significantly_different(self, position):
        return cp.linalg.norm(position - self.last_position) > 2500

    def reset_unused_points(self, position):
        self.reset_points(self.grass, position)
        self.reset_points(self.detail_meshes, position)
        self.reset_points(self.detail_prefabs, position)

    def reset_points(self, points, position):
        if points:
            for i in range(points.points.shape[0]):
                if cp.linalg.norm(position - points.points[i]) > self.radius_sqr:
                    points.points[i] = cp.zeros(3)
                    points.currently_used_points -= 1

    def calculate_avg_pos(self, position):
        avg_pos = cp.zeros(3)
        length = 0
        if self.generate_grass:
            length = min(1000, self.grass.number_of_points)
            avg_pos = cp.sum(self.grass.points[:length] - position, axis=0)
        elif self.generate_meshes:
            length = min(1000, self.detail_meshes.number_of_points)
            avg_pos = cp.sum(self.detail_meshes.points[:length] - position, axis=0)
        elif self.generate_prefabs:
            length = min(1000, self.detail_prefabs.number_of_points)
            avg_pos = cp.sum(self.detail_prefabs.points[:length] - position, axis=0)
        return avg_pos / length

    def should_generate_foliage(self):
        return ((self.generate_grass and self.grass.currently_used_points < self.grass.number_of_points) or
                (self.generate_meshes and self.detail_meshes.currently_used_points < self.detail_meshes.number_of_points) or
                (self.generate_prefabs and self.detail_prefabs.currently_used_points < self.detail_prefabs.number_of_points))

    def generate_foliage_raycast(self, avg_pos):
        pass

    def generate_foliage(self, avg_pos):
        pass

    def update_point_cloud(self):
        if self.generate_grass:
            self.point_cloud = self.grass.points

    def render_point_cloud(self):
        if self.generate_grass:
            # Replace with GPU draw call
            pass

import numpy as np
import cupy as cp

class FoliageExperimental:
    def __init__(self, planet):
        self.planet = planet
        self.max_points_per_frame = 1000
        self.size = 1500.0
        self.use_raycast = False
        self.radius_sqr = 2 * self.size * self.size

        self.grass = None
        self.detail_meshes = None
        self.detail_prefabs = None
        self.point_cloud = None
        self.matrices = None
        self.transforms = None

        self.last_position = cp.zeros(3)
        self.last_planet_position = cp.zeros(3)
        self.last_planet_rotation = cp.zeros(4)
        self.avg_pos = cp.zeros(3)

        self.generate_grass = False
        self.generate_meshes = False
        self.generate_prefabs = False

        self.initialized = False
        self.quad_size = 0
        self.quad_size_1 = 0

    class SurfacePoints:
        def __init__(self, number_of_points):
            self.number_of_points = number_of_points
            self.currently_used_points = 0
            self.points = cp.zeros((number_of_points, 3))
            self.normals = cp.zeros((number_of_points, 3))

    def awake(self):
        self.radius_sqr = 2 * self.size * self.size

    def initialize(self):
        self.generate_grass = self.planet.exp_grass
        self.generate_meshes = self.planet.exp_meshes
        self.generate_prefabs = self.planet.exp_prefabs

        self.quad_size = self.planet.quad_size
        self.quad_size_1 = self.quad_size - 1

        if self.generate_grass and self.planet.grass_per_quad > 0:
            self.grass = self.SurfacePoints(self.planet.grass_per_quad)
            self.point_cloud = cp.zeros((self.planet.grass_per_quad, 3))
        else:
            self.generate_grass = False

        number_detail_meshes = sum(mesh.number for mesh in self.planet.detail_meshes)
        if self.generate_meshes and number_detail_meshes > 0:
            self.detail_meshes = self.SurfacePoints(number_detail_meshes)
            self.matrices = cp.zeros((number_detail_meshes, 4, 4))
        else:
            self.generate_meshes = False

        number_detail_prefabs = sum(prefab.number for prefab in self.planet.detail_prefabs)
        if self.generate_prefabs and number_detail_prefabs > 0:
            self.detail_prefabs = self.SurfacePoints(number_detail_prefabs)
            self.transforms = [None] * number_detail_prefabs
        else:
            self.generate_prefabs = False

        offset = 0
        for prefab in self.planet.detail_prefabs:
            for j in range(prefab.number):
                self.transforms[j + offset] = prefab.prefab
            offset += prefab.number

        self.initialized = True

    def reset(self):
        self.grass = None
        self.detail_meshes = None
        self.detail_prefabs = None
        self.matrices = None
        self.transforms = None
        self.initialized = False

    def update_meshes(self):
        offset = 0
        rotation = self.calculate_rotation(self.detail_meshes.points[0])
        for mesh in self.planet.detail_meshes:
            for j in range(mesh.number):
                if not cp.allclose(self.detail_meshes.points[j + offset], 0):
                    self.matrices[j + offset] = self.calculate_matrix(self.detail_meshes.points[j + offset], mesh.mesh_offset_up, self.detail_meshes.normals[j + offset], rotation, mesh.mesh_scale)
            offset += mesh.number

    def render_meshes(self):
        offset = 0
        for mesh in self.planet.detail_meshes:
            if mesh.use_gpu_instancing:
                for j in range(mesh.number):
                    if not cp.allclose(self.detail_meshes.points[j + offset], 0):
                        # Replace with GPU draw call
                        pass
            else:
                to_render = [self.matrices[j + offset] for j in range(mesh.number) if not cp.allclose(self.detail_meshes.points[j + offset], 0)]
                # Replace with instanced draw call
            offset += mesh.number

    def update_transforms(self):
        offset = 0
        rotation = self.calculate_rotation(self.detail_prefabs.points[0])
        for prefab in self.planet.detail_prefabs:
            for j in range(prefab.number):
                if not cp.allclose(self.detail_prefabs.points[j + offset], 0):
                    self.transforms[j + offset].position = self.detail_prefabs.points[j + offset] + prefab.mesh_offset_up * self.detail_prefabs.normals[j + offset]
                    self.transforms[j + offset].rotation = rotation
                    self.transforms[j + offset].set_active(True)
                else:
                    self.transforms[j + offset].set_active(False)
            offset += prefab.number

    def update(self):
        tr_position = cp.asarray(self.get_position())
        down = self.calculate_down_direction(tr_position)
        hit = self.check_raycast(tr_position, down)
        if not hit:
            if self.initialized:
                self.reset()
            return
        if hit.tag == "Quad":
            biome = self.planet.find_quad(int(hit.name[5:])).biome
            if not self.is_biome_included(biome):
                return
            if not self.initialized:
                self.initialize()

        self.update_position_and_rotation(tr_position, down)

        if self.is_position_significantly_different(tr_position):
            self.last_position = tr_position
            self.reset_unused_points(tr_position)
            self.avg_pos = self.calculate_avg_pos(tr_position)

        if self.should_generate_foliage():
            if self.use_raycast:
                self.generate_foliage_raycast(self.avg_pos)
            else:
                self.generate_foliage(self.avg_pos)
            self.update_point_cloud()
            self.update_meshes()
            self.update_transforms()
        self.render_point_cloud()
        self.render_meshes()

    def calculate_rotation(self, point):
        return cp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    def calculate_matrix(self, point, offset, normal, rotation, scale):
        return cp.eye(4)

    def get_position(self):
        return cp.array([0, 0, 0])

    def calculate_down_direction(self, position):
        return cp.array([0, -1, 0])

    def check_raycast(self, position, direction):
        return None

    def is_biome_included(self, biome):
        return True

    def update_position_and_rotation(self, position, direction):
        delta_pos = self.planet.position - self.last_planet_position
        self.last_position -= delta_pos
        self.update_points(self.grass, delta_pos)
        self.update_points(self.detail_meshes, delta_pos)
        self.update_points(self.detail_prefabs, delta_pos)
        self.last_planet_position = self.planet.position

        rotate = self.calculate_rotation_difference()
        self.rotate_points(self.grass, rotate)
        self.rotate_points(self.detail_meshes, rotate)
        self.rotate_points(self.detail_prefabs, rotate)
        self.last_planet_rotation = self.planet.rotation

    def update_points(self, points, delta_pos):
        if points:
            for i in range(points.points.shape[0]):
                if not cp.allclose(points.points[i], 0):
                    points.points[i] += delta_pos

    def calculate_rotation_difference(self):
        return cp.array([1, 0, 0, 0])

    def rotate_points(self, points, rotate):
        if points:
            for i in range(points.points.shape[0]):
                if not cp.allclose(points.points[i], 0):
                    points.points[i] = self.rotate_point(points.points[i], rotate)

    def rotate_point(self, point, rotate):
        return point

    def is_position_significantly_different(self, position):
        return cp.linalg.norm(position - self.last_position) > 2500

    def reset_unused_points(self, position):
        self.reset_points(self.grass, position)
        self.reset_points(self.detail_meshes, position)
        self.reset_points(self.detail_prefabs, position)

    def reset_points(self, points, position):
        if points:
            for i in range(points.points.shape[0]):
                if cp.linalg.norm(position - points.points[i]) > self.radius_sqr:
                    points.points[i] = cp.zeros(3)
                    points.currently_used_points -= 1

    def calculate_avg_pos(self, position):
        avg_pos = cp.zeros(3)
        length = 0
        if self.generate_grass:
            length = min(1000, self.grass.number_of_points)
            avg_pos = cp.sum(self.grass.points[:length] - position, axis=0)
        elif self.generate_meshes:
            length = min(1000, self.detail_meshes.number_of_points)
            avg_pos = cp.sum(self.detail_meshes.points[:length] - position, axis=0)
        elif self.generate_prefabs:
            length = min(1000, self.detail_prefabs.number_of_points)
            avg_pos = cp.sum(self.detail_prefabs.points[:length] - position, axis=0)
        return avg_pos / length

    def should_generate_foliage(self):
        return ((self.generate_grass and self.grass.currently_used_points < self.grass.number_of_points) or
                (self.generate_meshes and self.detail_meshes.currently_used_points < self.detail_meshes.number_of_points) or
                (self.generate_prefabs and self.detail_prefabs.currently_used_points < self.detail_prefabs.number_of_points))

    def generate_foliage_raycast(self, avg_pos):
        pass

    def generate_foliage(self, avg_pos):
        pass

    def update_point_cloud(self):
        if self.generate_grass:
            self.point_cloud = self.grass.points

    def render_point_cloud(self):
        if self.generate_grass:
            # Replace with GPU draw call
            pass

import numpy as np
import cupy as cp

class FoliageExperimental:
    # ... (Other methods and initialization)

    def generate_foliage(self, avg_pos):
        tr_position = cp.asarray(self.get_position())
        down = self.calculate_down_direction(tr_position)

        mat = self.calculate_transformation_matrix(down, avg_pos)

        avg_pos = self.transform_to_local_space(avg_pos, mat)
        ray_points = self.calculate_ray_points(avg_pos, down)
        hit_points, hit_quads = self.perform_raycast(tr_position, down, ray_points)

        if not hit_points:
            return

        vertices, normals = self.extract_vertices_and_normals(hit_quads)
        pot_triangles = self.calculate_potential_triangles(hit_quads, hit_points, vertices, normals, mat)

        if not self.are_triangles_valid(pot_triangles):
            return

        current_frame_points = 0
        counter, counter_reset = 0, len(pot_triangles) - 1

        while not pot_triangles[counter]:
            counter = (counter + 1) % counter_reset

        if self.generate_grass:
            self.generate_surface_points(self.grass, pot_triangles, vertices, normals, counter, counter_reset, current_frame_points)

        if self.generate_meshes:
            self.generate_surface_points(self.detail_meshes, pot_triangles, vertices, normals, counter, counter_reset, current_frame_points)

        if self.generate_prefabs:
            self.generate_surface_points(self.detail_prefabs, pot_triangles, vertices, normals, counter, counter_reset, current_frame_points)

    def calculate_transformation_matrix(self, down, avg_pos):
        largest = cp.abs(down).max()
        index = cp.argmax(cp.abs(down))
        a, b = self.calculate_basis_vectors(down, index)
        return cp.array([[a[0], down[0], b[0], 0], [a[1], down[1], b[1], 0], [a[2], down[2], b[2], 0], [0, 0, 0, 1]])

    def transform_to_local_space(self, avg_pos, mat):
        avg_pos = mat @ avg_pos
        avg_pos[1] = 0
        return -cp.linalg.inv(mat) @ avg_pos

    def calculate_ray_points(self, avg_pos, down):
        a, b = self.calculate_basis_vectors(down, cp.argmax(cp.abs(down)))
        ray0 = a * -self.size + b * -self.size
        ray1 = a * -self.size + b * self.size
        ray2 = a * self.size + b * self.size
        ray3 = a * self.size + b * -self.size
        if cp.linalg.norm(avg_pos) > 0.2:
            perpendicular = cp.cross(avg_pos, down)
            ray0, ray1, ray2, ray3 = self.adjust_ray_points(avg_pos, perpendicular)
        return [ray0, ray1, ray2, ray3]

    def perform_raycast(self, tr_position, down, ray_points):
        hit_points = []
        hit_quads = []
        for ray in ray_points:
            hit = self.raycast(tr_position + ray + down * -500, down)
            if hit and hit.tag == "Quad":
                hit_points.append(hit.point)
                hit_quads.append(self.planet.find_quad(int(hit.name[5:])))
        return hit_points, hit_quads

    def extract_vertices_and_normals(self, hit_quads):
        vertices = [quad.mesh.vertices for quad in hit_quads]
        normals = [quad.mesh.normals for quad in hit_quads]
        return vertices, normals

    def calculate_potential_triangles(self, hit_quads, hit_points, vertices, normals, mat):
        pot_triangles = [[] for _ in hit_quads]
        for i, quad in enumerate(hit_quads):
            to_quad, lower_limits, upper_limits = self.calculate_limits_and_transformation(quad, hit_points[i], mat)
            triangles = quad.mesh.triangles
            for j in range(0, len(triangles), 3):
                f, g, h = triangles[j], triangles[j + 1], triangles[j + 2]
                if self.is_within_limits(f, g, h, lower_limits, upper_limits, to_quad):
                    pot_triangles[i].extend([f, g, h])
        return pot_triangles

    def are_triangles_valid(self, pot_triangles):
        return any(triangles for triangles in pot_triangles)

    def generate_surface_points(self, points, pot_triangles, vertices, normals, counter, counter_reset, current_frame_points):
        for i in range(points.points.shape[0]):
            if cp.allclose(points.points[i], 0):
                index = cp.random.choice(pot_triangles[counter])
                index -= index % 3
                tri_a, tri_b, tri_c = vertices[counter][pot_triangles[counter][index]], vertices[counter][pot_triangles[counter][index + 1]], vertices[counter][pot_triangles[counter][index + 2]]
                x, y = cp.random.rand(), cp.random.rand()
                if x + y >= 1:
                    x, y = 1 - x, 1 - y
                points.points[i] = hit_quads[counter].rendered_quad.transform_point(tri_a + x * (tri_b - tri_a) + y * (tri_c - tri_a))
                points.normals[i] = hit_quads[counter].rendered_quad.transform_direction(normals[counter][pot_triangles[counter][index]])
                counter = (counter + 1) % counter_reset
                current_frame_points += 1
                if current_frame_points > self.max_points_per_frame:
                    break
        points.currently_used_points += current_frame_points

    # Additional helper methods go here...

import cupy as cp

class FoliageExperimental:
    # ... (Other methods and initialization)

    def generate_foliage_raycast(self, avg_pos):
        tr_position = cp.asarray(self.get_position())
        down = cp.asarray(self.planet.get_position() - tr_position).normalized()

        largest = cp.abs(down).max()
        index = cp.argmax(cp.abs(down))

        if index == 0:
            a = cp.array([-down[1] / down[0], 1, 0]).normalized()
        elif index == 1:
            a = cp.array([1, -down[0] / down[1], 0]).normalized()
        else:
            if down[2] > 0:
                b = cp.array([1, 0, -down[0] / down[2]]).normalized()
                a = -cp.cross(b, down)
            else:
                b = -cp.array([1, 0, -down[0] / down[2]]).normalized()
                a = cp.cross(-b, down)
        
        b = cp.cross(a, down)
        mat = cp.array([[a[0], down[0], b[0], 0], [a[1], down[1], b[1], 0], [a[2], down[2], b[2], 0], [0, 0, 0, 1]])

        avg_pos = mat @ avg_pos
        avg_pos[1] = 0
        avg_pos = -cp.linalg.inv(mat) @ avg_pos

        mag = cp.linalg.norm(avg_pos)
        avg_pos /= mag
        mag /= self.size

        perpendicular = cp.cross(down, avg_pos)
        current_frame_points = 0

        if self.generate_grass:
            self.generate_grass_points(avg_pos, down, a, b, perpendicular, mag, tr_position, current_frame_points)
        
        if self.generate_meshes:
            self.generate_mesh_points(avg_pos, down, a, b, perpendicular, mag, tr_position, current_frame_points)

        if self.generate_prefabs:
            self.generate_prefab_points(avg_pos, down, a, b, perpendicular, mag, tr_position, current_frame_points)

    def generate_grass_points(self, avg_pos, down, a, b, perpendicular, mag, tr_position, current_frame_points):
        for i in range(self.grass.points.shape[0]):
            if cp.allclose(self.grass.points[i], 0):
                if current_frame_points > self.max_points_per_frame:
                    break

                if mag > 0.2:
                    ray_origin = tr_position + avg_pos * cp.random.uniform(self.size / 2, self.size) + perpendicular * cp.random.uniform(-self.size, self.size) - 500 * down
                else:
                    ray_origin = tr_position + a * cp.random.uniform(-self.size, self.size) + b * cp.random.uniform(-self.size, self.size) - 500 * down

                hit = self.raycast(ray_origin, down)
                if hit and hit.tag == "Quad":
                    self.grass.points[i] = hit.point
                    self.grass.normals[i] = hit.normal
                    current_frame_points += 1

        self.grass.currently_used_points += current_frame_points

    def generate_mesh_points(self, avg_pos, down, a, b, perpendicular, mag, tr_position, current_frame_points):
        for i in range(self.detail_meshes.points.shape[0]):
            if cp.allclose(self.detail_meshes.points[i], 0):
                if current_frame_points > self.max_points_per_frame:
                    break

                if mag > 0.2:
                    ray_origin = tr_position + avg_pos * cp.random.uniform(self.size / 2, self.size) + perpendicular * cp.random.uniform(-self.size, self.size) - 500 * down
                else:
                    ray_origin = tr_position + a * cp.random.uniform(-self.size, self.size) + b * cp.random.uniform(-self.size, self.size) - 500 * down

                hit = self.raycast(ray_origin, down)
                if hit and hit.tag == "Quad":
                    self.detail_meshes.points[i] = hit.point
                    self.detail_meshes.normals[i] = hit.normal
                    current_frame_points += 1

        self.detail_meshes.currently_used_points += current_frame_points

    def generate_prefab_points(self, avg_pos, down, a, b, perpendicular, mag, tr_position, current_frame_points):
        for i in range(self.detail_prefabs.points.shape[0]):
            if cp.allclose(self.detail_prefabs.points[i], 0):
                if current_frame_points > self.max_points_per_frame:
                    break

                if mag > 0.2:
                    ray_origin = tr_position + avg_pos * cp.random.uniform(self.size / 2, self.size) + perpendicular * cp.random.uniform(-self.size, self.size) - 500 * down
                else:
                    ray_origin = tr_position + a * cp.random.uniform(-self.size, self.size) + b * cp.random.uniform(-self.size, self.size) - 500 * down

                hit = self.raycast(ray_origin, down)
                if hit and hit.tag == "Quad":
                    self.detail_prefabs.points[i] = hit.point
                    self.detail_prefabs.normals[i] = hit.normal
                    current_frame_points += 1

        self.detail_prefabs.currently_used_points += current_frame_points

import numpy as np

class ComputeShader:
    def __init__(self):
        self.textures = {}

    def find_kernel(self, kernel_name):
        # Placeholder for kernel ID retrieval
        return 0

    def set_texture(self, kernel_id, variable_name, texture):
        self.textures[variable_name] = texture

class Texture2D:
    def __init__(self, wrap_mode=None):
        self.wrap_mode = wrap_mode

class LoadHeightmapComputeShader:
    def __init__(self, compute_shader, heightmaps, variable_names):
        self.compute_shader = compute_shader
        self.heightmaps = heightmaps
        self.variable_names = variable_names

    def awake(self):
        if len(self.heightmaps) != len(self.variable_names):
            raise IndexError("Heightmaps and variableNames must have the same length!")

        kernel_id = self.compute_shader.find_kernel("ComputePositions")

        for i in range(len(self.heightmaps)):
            self.heightmaps[i].wrap_mode = 'Repeat'
            self.compute_shader.set_texture(kernel_id, self.variable_names[i], self.heightmaps[i])

class Planet:
    def __init__(self, radius):
        self.radius = radius

class Transform:
    def __init__(self, position):
        self.position = position

class Rigidbody:
    def __init__(self, mass):
        self.mass = mass

    def add_force(self, force):
        pass  # Placeholder for adding force to the object

class SphericalGravity:
    def __init__(self, objects, acceleration, planet):
        self.objects = objects
        self.acceleration = acceleration
        self.radius = planet.radius

    def start(self):
        self.radius = self.radius

    def fixed_update(self):
        for obj in self.objects:
            obj_pos = obj['transform'].position
            distance_vector = self.planet_position - obj_pos
            distance = np.linalg.norm(distance_vector)
            force_direction = distance_vector / distance
            force_magnitude = -self.acceleration * obj['rigidbody'].mass * self.radius**2 / distance**2
            obj['rigidbody'].add_force(force_direction * force_magnitude)

class MathFunctions:
    @staticmethod
    def lat_lon_to_xyz(latlon, radius):
        lat, lon = np.deg2rad(latlon[0]), np.deg2rad(latlon[1])
        x = radius * np.cos(lat) * np.cos(lon)
        y = radius * np.cos(lat) * np.sin(lon)
        z = radius * np.sin(lat)
        return np.array([x, y, z])

    @staticmethod
    def xyz_to_lat_lon(xyz):
        x, y, z = xyz
        lat = np.rad2deg(np.arctan2(z, np.sqrt(x**2 + y**2)))
        lon = np.rad2deg(np.arctan2(y, x))
        return np.array([lat, lon])

class Coordinates:
    def __init__(self, latlon, planet, teleport, height):
        self.latlon = latlon
        self.planet = planet
        self.teleport = teleport
        self.height = height

    def start(self):
        if self.teleport:
            self.planet.position = -MathFunctions.lat_lon_to_xyz(self.latlon, self.planet.radius) * self.height

    def update(self):
        relative_planet = self.position - self.planet.position
        self.latlon = MathFunctions.xyz_to_lat_lon(np.linalg.inv(self.planet.rotation_matrix) @ relative_planet)

import numpy as np

class ModuleWrapper:
    def __init__(self, m):
        self.m = m

class ComputeShaderGenerator:
    @staticmethod
    def generate_compute_shader(module):
        return "compute shader code"  # Placeholder for the actual compute shader generation

class Utils:
    @staticmethod
    def generate_preview(module, width, height):
        return np.zeros((width, height))  # Placeholder for the preview generation

class Node:
    def get_input_value(self, input_name, default):
        # Placeholder for getting the input value
        return ModuleWrapper(None)

class SavingNode(Node):
    def __init__(self):
        self.input = None
        self.filename = "noiseModule"
        self.preview = None

    def serialize(self):
        module = self.get_input_value("input", None).m
        with open(f"{self.filename}.bytes", "wb") as f:
            f.write(module.serialize())  # Assuming the module has a serialize method

    def serialize_compute_shader(self):
        module = self.get_input_value("input", None).m
        with open(f"{self.filename}.compute", "w") as f:
            f.write(ComputeShaderGenerator.generate_compute_shader(module))

    def generate_preview(self):
        module = self.get_input_value("input", None).m
        self.preview = Utils.generate_preview(module, 256, 256)

class CameraMove:
    def __init__(self, transform, speed=10.0, fast_speed=1000.0, rot_speed=0.25):
        self.transform = transform
        self.speed = speed
        self.fast_speed = fast_speed
        self.rot_speed = rot_speed
        self.last_mouse_pos = np.array([0, 0])

    def fixed_update(self, input):
        self.transform.translate(0, 0, input.get_axis("Vertical") * (self.speed + (input.get_axis("Fire3") * self.fast_speed)))

        delta = self.last_mouse_pos - input.mouse_position

        if input.get_mouse_button(0):
            self.transform.rotate([-delta[1] * self.rot_speed, -delta[0] * self.rot_speed, 0])

        self.transform.rotate([0, 0, -input.get_axis("Horizontal")])

        self.last_mouse_pos = input.mouse_position

class FloatingOrigin:
    def __init__(self, distance_from_original_origin):
        self.distance_from_original_origin = distance_from_original_origin

class ScaledSpace:
    def __init__(self, main_camera, scaled_space_camera, scale_factor=100000.0):
        self.scale_factor = scale_factor
        self.main_camera = main_camera
        self.scaled_space_camera = scaled_space_camera
        self.fo = None

    def start(self, fo):
        self.fo = fo

    def update(self):
        if self.fo:
            self.scaled_space_camera.position = (self.main_camera.position + self.fo.distance_from_original_origin) / self.scale_factor
        else:
            self.scaled_space_camera.position = self.main_camera.position / self.scale_factor
        self.scaled_space_camera.rotation = self.main_camera.rotation

class Fade:
    def __init__(self, planet, main_cam, fade_start, fade_end):
        self.planet = planet
        self.main_cam = main_cam
        self.fade_start = fade_start
        self.fade_end = fade_end

    def update(self):
        inv_diff = 1 / (self.fade_end - self.fade_start)
        fade = np.clip((self.fade_end - np.linalg.norm(self.planet.position - self.main_cam.position)) * inv_diff, 0, 1)

        renderer = self.get_renderer()
        color = renderer.get_shared_material().color
        color.a = 1.0 - fade
        renderer.get_shared_material().color = color

    def get_renderer(self):
        # Placeholder for getting the renderer
        class Renderer:
            class SharedMaterial:
                def __init__(self):
                    self.color = type('Color', (object,), {'a': 1.0})

            def get_shared_material(self):
                return Renderer.SharedMaterial()

        return Renderer()

# Placeholder classes for Transform and Input
class Transform:
    def __init__(self, position, rotation):
        self.position = position
        self.rotation = rotation

    def translate(self, x, y, z):
        pass  # Implement translation logic

    def rotate(self, rotation_vector):
        pass  # Implement rotation logic

class Input:
    @staticmethod
    def get_axis(axis_name):
        return 0.0  # Placeholder for axis value

    @staticmethod
    def get_mouse_button(button_index):
        return False  # Placeholder for mouse button state

    @staticmethod
    def mouse_position():
        return np.array([0, 0])  # Placeholder for mouse position

import numpy as np

class Material:
    def __init__(self):
        self.texture_offsets = {}

    def set_texture_offset(self, texture_name, offset):
        self.texture_offsets[texture_name] = offset

class WaterFlow:
    def __init__(self, water, flow_speed):
        self.water = water
        self.flow_speed = flow_speed
        self.flow_vector = np.array([0.0, 0.0])

    def update(self):
        self.flow_vector[0] += (self.flow_speed / 100.0)
        self.flow_vector[1] -= (self.flow_speed / 100.0)

        self.water.set_texture_offset("_Normals", -self.flow_vector)
        self.water.set_texture_offset("_ReflectTex", self.flow_vector)
        self.water.set_texture_offset("_WaveMap", self.flow_vector)



class Module:
    def __init__(self, data):
        self.data = data

class Const(Module):
    def __init__(self, value):
        super().__init__(value)

class ModuleWrapper:
    def __init__(self, module):
        self.module = module

class PTNode:
    def __init__(self):
        self.output = None

    def get_value(self, port):
        return ModuleWrapper(self.get_module())

    def get_module(self):
        return None

class FromSavedNode(PTNode):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.serialized = None

    def get_value(self, port):
        return ModuleWrapper(self.get_module())

    def get_module(self):
        if self.serialized is None:
            return Const(0)
        return self.serialized

class PropertyAttribute:
    pass

class NodeEnumAttribute(PropertyAttribute):
    pass

# Placeholder classes for Transform and Quaternion
class Transform:
    def __init__(self, position):
        self.position = position

class Quaternion:
    @staticmethod
    def euler(rotation_speed):
        # Convert Euler angles to a rotation matrix or similar representation
        return rotation_speed

# Example usage:
water_material = Material()
water_flow = WaterFlow(water_material, 1.0)
water_flow.update()

parent_planet_transform = Transform(np.array([0, 0, 0]))
object_transform = Transform(np.array([1, 0, 0]))


import threading
import numpy as np
from PIL import Image

class NodeEditor:
    def on_body_gui(self):
        self.on_body_gui_light()

    def on_body_gui_light(self):
        pass

class EditorGUILayout:
    @staticmethod
    def toggle(label, value):
        return value  # Placeholder for a toggle switch

    @staticmethod
    def text_field(label, value):
        return value  # Placeholder for a text field

class GUILayout:
    @staticmethod
    def label(content):
        print(content)  # Placeholder for GUI label

    @staticmethod
    def button(label):
        print(f"Button: {label}")
        return True  # Always return True for simplicity

class NodeEditorWindow:
    current = None

    @staticmethod
    def repaint():
        print("Repainting window")  # Placeholder for repainting the editor window

class Texture2D:
    def __init__(self, image=None):
        self.image = image

    def get_texture_2d(self):
        return self.image

class Utils:
    @staticmethod
    def generate_preview_heightmap(module, width, height):
        # Placeholder for generating a heightmap
        return Texture2D(np.zeros((width, height)))

class PreviewNode:
    def __init__(self):
        self.preview_changed = False
        self.preview_heightmap = None
        self.preview = None
        self.auto_update_preview = True

    def update_preview(self):
        module = self.get_module()
        if module is not None:
            self.preview_heightmap = Utils.generate_preview_heightmap(module, 128, 128)
            self.preview_changed = True

    def get_module(self):
        return None  # Placeholder for module retrieval

class PreviewNodeEditor(NodeEditor):
    def on_body_gui(self):
        super().on_body_gui_light()

        node = self.target

        if GUILayout.button("Generate Preview"):
            threading.Thread(target=node.update_preview).start()

        if node.preview_changed:
            if node.preview_heightmap is None:
                return
            node.preview = node.preview_heightmap.get_texture_2d()
            node.preview_changed = False
            NodeEditorWindow.repaint()

        GUILayout.label(node.preview)
        node.auto_update_preview = EditorGUILayout.toggle("Auto-update", node.auto_update_preview)

class Module:
    def __init__(self, data):
        self.data = data

class Const(Module):
    def __init__(self, value):
        super().__init__(value)

class PTNode:
    def __init__(self):
        self.preview_changed = False
        self.preview_heightmap = None
        self.preview = None
        self.auto_update_preview = True

    def get_module(self):
        return None  # Placeholder for module retrieval

    def update_preview(self):
        module = self.get_module()
        if module is not None:
            self.preview_heightmap = Utils.generate_preview_heightmap(module, 128, 128)
            self.preview_changed = True

class ModuleWrapper:
    def __init__(self, module):
        self.module = module

    @staticmethod
    def zero():
        return ModuleWrapper(Const(0))

class SavingNode(PTNode):
    def __init__(self):
        super().__init__()
        self.filename = "noiseModule"

    def serialize(self):
        module = self.get_module()
        with open(f"{self.filename}.bytes", "wb") as f:
            f.write(module.data)  # Assuming the module has a data attribute

    def serialize_compute_shader(self):
        module = self.get_module()
        with open(f"{self.filename}.compute", "w") as f:
            f.write("compute shader code")  # Placeholder for actual compute shader code

    def generate_preview(self):
        module = self.get_module()
        if module is not None:
            self.preview = Utils.generate_preview_heightmap(module, 256, 256).get_texture_2d()

class SavingNodeEditor(NodeEditor):
    def on_body_gui(self):
        node = self.target
        super().on_body_gui_light()

        node.filename = EditorGUILayout.text_field("Filename", node.filename)

        if GUILayout.button("Serialize"):
            node.serialize()
            # AssetDatabase.refresh()  # Placeholder for refreshing the asset database

        if GUILayout.button("Serialize Compute Shader"):
            node.serialize_compute_shader()
            # AssetDatabase.refresh()  # Placeholder for refreshing the asset database

        if GUILayout.button("Generate Preview"):
            node.generate_preview()

        GUILayout.label(node.preview)

import numpy as np
from typing import List

class Vector3d:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class QuaternionD:
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    @staticmethod
    def euler(rotation_speed):
        # Placeholder for Euler angles to quaternion conversion
        return QuaternionD(0, 0, 0, 1)

class Vector2Int:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class ExtensionMethods:
    @staticmethod
    def to_vector3d(v3):
        return Vector3d(v3[0], v3[1], v3[2])

    @staticmethod
    def to_quaterniond(q):
        return QuaternionD(q[0], q[1], q[2], q[3])

    @staticmethod
    def round_to_int(v):
        return Vector2Int(int(round(v[0])), int(round(v[1])))

    @staticmethod
    def values_to_string(array):
        return "{" + ", ".join(f"{value:.6f}" for value in array) + "}"

class Planet:
    def __init__(self):
        self.rotation = QuaternionD(0, 0, 0, 1)

class Transform:
    def __init__(self):
        self.children = []
        self.rotation = QuaternionD(0, 0, 0, 1)

    def add_child(self, child):
        self.children.append(child)

    def get_component(self, component_type):
        if component_type == Planet:
            return Planet()
        return None

class MonoBehaviour:
    pass


class TextAsset:
    def __init__(self, text):
        self.text = text

class Module:
    def __init__(self, data):
        self.data = data

class Const(Module):
    def __init__(self, value):
        super().__init__(value)

class HeightmapModule(Module):
    def __init__(self, text_asset_bytes, compute_shader_name, use_bicubic_interpolation):
        self.text_asset_bytes = text_asset_bytes
        self.compute_shader_name = compute_shader_name
        self.use_bicubic_interpolation = use_bicubic_interpolation

    def init(self):
        pass  # Placeholder for initialization logic

class ModuleWrapper:
    def __init__(self, module):
        self.module = module

    @staticmethod
    def zero():
        return ModuleWrapper(Const(0))

class PTNode:
    def __init__(self):
        self.output = None

    def get_value(self, port):
        if port != "output":
            return None
        return ModuleWrapper(self.get_module())

    def get_module(self):
        return None

class HeightmapNode(PTNode):
    def __init__(self):
        super().__init__()
        self.heightmap_text_asset = None
        self.text_asset_bytes = None
        self.compute_shader_name = None
        self.use_bicubic_interpolation = False

    def get_value(self, port):
        if port != "output":
            return None
        return ModuleWrapper(self.get_module())

    def get_module(self):
        if self.text_asset_bytes is None and self.compute_shader_name is None:
            return Const(-1)
        heightmap_module = HeightmapModule(self.text_asset_bytes, self.compute_shader_name, self.use_bicubic_interpolation)
        heightmap_module.init()
        return heightmap_module




heightmap_node = HeightmapNode()
module = heightmap_node.get_module()

import numpy as np
from typing import List

class EditorWindow:
    def on_gui(self):
        pass

class GUILayout:
    @staticmethod
    def label(content, style=None):
        print(content)  # Placeholder for GUI label

    @staticmethod
    def button(label):
        print(f"Button: {label}")
        return True  # Always return True for simplicity

class EditorGUILayout:
    @staticmethod
    def object_field(label, obj, obj_type, allow_scene_objects):
        return obj  # Placeholder for object field

    @staticmethod
    def float_field(label, value):
        return value  # Placeholder for float field

    @staticmethod
    def text_field(label, value):
        return value  # Placeholder for text field

    @staticmethod
    def enum_popup(label, enum_value):
        return enum_value  # Placeholder for enum popup

class Planet:
    def __init__(self):
        self.detail_distances = [0.0] * 10  # Placeholder for detail distances

class GameObject:
    @staticmethod
    def find(name):
        return GameObject()

    def get_component(self, component_type):
        return Planet()  # Placeholder for getting a component

class DetailDistancesCalc(EditorWindow):
    def __init__(self):
        self.start_value = 20000
        self.planet_name = "Planet"
        self.planet = None

    @staticmethod
    def init():
        window = DetailDistancesCalc()
        window.on_gui()

    def on_gui(self):
        try:
            if not self.planet:
                self.planet = GameObject.find(self.planet_name).get_component(Planet)
        except:
            pass
        GUILayout.label("Detail Distances Calculator")
        self.planet = EditorGUILayout.object_field("Planet", self.planet, Planet, True)
        self.start_value = EditorGUILayout.float_field("start value", self.start_value)

        if GUILayout.button("Calculate"):
            detail_distances = self.planet.detail_distances
            for i in range(len(detail_distances)):
                if i == 0:
                    detail_distances[i] = self.start_value
                else:
                    detail_distances[i] = detail_distances[i - 1] / 2.0

class Texture2D:
    def __init__(self, width, height):
        self.width = width
        self.height = height

class Heightmap:
    def __init__(self, texture, is_grayscale, channel):
        self.texture = texture
        self.is_grayscale = is_grayscale
        self.channel = channel

    def get_file_bytes(self):
        return bytearray()  # Placeholder for file bytes

class Channel:
    R, G, B, A, Gray = range(5)


class Utils:
    @staticmethod
    def deserialize_text_asset(text_asset):
        return Module()  # Placeholder for deserialization

class Debug:
    @staticmethod
    def log(message):
        print(message)

    @staticmethod
    def log_error(message):
        print(f"Error: {message}")

class Module:
    def __init__(self):
        pass

class TextAsset:
    def __init__(self, text):
        self.text = text

class NodeEditor:
    def on_body_gui(self):
        self.on_body_gui_light()

    def on_body_gui_light(self):
        pass

class GUIStyle:
    def __init__(self):
        self.alignment = None

class TextAnchor:
    UpperCenter = None

class NodeEditorWindow:
    current = None

    @staticmethod
    def repaint():
        print("Repainting window")  # Placeholder for repainting the editor window

class FromSavedNode:
    def __init__(self):
        self.serialized = None
        self.preview_changed = False
        self.preview_heightmap = None
        self.preview = None

    def update_preview(self):
        pass  # Placeholder for updating preview

class FromSavedNodeEditor(NodeEditor):
    def __init__(self):
        self.module_ta = None

    def on_body_gui(self):
        super().on_body_gui_light()
        node = self.target

        self.module_ta = EditorGUILayout.object_field("Noise", self.module_ta, TextAsset, False)
        if EditorGUILayout.end_change_check():
            try:
                node.serialized = Utils.deserialize_text_asset(self.module_ta)
            except:
                Debug.log_error("Cannot deserialize. Invalid Noise Module!")

        if node.preview_changed:
            if node.preview_heightmap is None:
                return
            node.preview = node.preview_heightmap.get_texture_2d()
            node.preview_changed = False
            NodeEditorWindow.repaint()

        if node.preview is None:
            threading.Thread(target=node.update_preview).start()

        centered = GUIStyle()
        centered.alignment = TextAnchor.UpperCenter

        GUILayout.label(node.preview, centered)

# Placeholder method to simulate EditorGUILayout.EndChangeCheck()
def editor_gui_end_change_check():
    return True

EditorGUILayout.end_change_check = editor_gui_end_change_check



import numpy as np
from PIL import Image
import threading

class TextAsset:
    def __init__(self, text):
        self.text = text
        self.bytes = text.encode()

class EditorGUI:
    @staticmethod
    def begin_change_check():
        pass

    @staticmethod
    def end_change_check():
        return True  # Always return True for simplicity

class GUILayout:
    @staticmethod
    def label(content, style=None):
        print(content)  # Placeholder for GUI label

    @staticmethod
    def button(label):
        print(f"Button: {label}")
        return True  # Always return True for simplicity

class EditorGUILayout:
    @staticmethod
    def object_field(label, obj, obj_type, allow_scene_objects):
        return obj  # Placeholder for object field

    @staticmethod
    def text_field(label, value):
        return value  # Placeholder for text field

    @staticmethod
    def toggle(label, value):
        return value  # Placeholder for toggle switch

class NodeEditorWindow:
    current = None

    @staticmethod
    def repaint():
        print("Repainting window")  # Placeholder for repainting the editor window

class Texture2D:
    def __init__(self, width, height, image=None):
        self.width = width
        self.height = height
        self.image = image if image is not None else Image.new('L', (width, height))

    def get_texture_2d(self):
        return self.image

class GUIStyle:
    def __init__(self):
        self.alignment = None

class TextAnchor:
    UpperCenter = None

class NodeEditor:
    def on_body_gui(self):
        self.on_body_gui_light()

    def on_body_gui_light(self):
        pass

class HeightmapNode:
    def __init__(self):
        self.heightmap_text_asset = None
        self.text_asset_bytes = None
        self.compute_shader_name = ""
        self.use_bicubic_interpolation = False
        self.preview = None
        self.preview_changed = False
        self.preview_heightmap = None

    def update_preview(self):
        pass  # Placeholder for updating preview

class HeightmapNodeEditor(NodeEditor):
    def on_body_gui(self):
        node = self.target
        super().on_body_gui_light()

        EditorGUI.begin_change_check()
        node.heightmap_text_asset = EditorGUILayout.object_field("Heightmap", node.heightmap_text_asset, TextAsset, False)
        if node.heightmap_text_asset is not None and (EditorGUI.end_change_check() or node.text_asset_bytes is None):
            node.text_asset_bytes = node.heightmap_text_asset.bytes

        node.compute_shader_name = EditorGUILayout.text_field("Name", node.compute_shader_name)
        node.use_bicubic_interpolation = EditorGUILayout.toggle("Use Bicubic Interpolation", node.use_bicubic_interpolation)

        if node.preview is None:
            node.update_preview()

        if node.preview_changed:
            if node.preview_heightmap is None:
                return
            node.preview = node.preview_heightmap.get_texture_2d()
            node.preview_changed = False
            NodeEditorWindow.repaint()

        centered = GUIStyle()
        centered.alignment = TextAnchor.UpperCenter

        GUILayout.label(node.preview, centered)

class Module:
    def __init__(self):
        pass

    def get_noise(self, x, y, z):
        return 0.0  # Placeholder for noise generation

class Heightmap:
    def __init__(self, width, height, is_grayscale, use_bicubic_interpolation):
        self.width = width
        self.height = height
        self.is_grayscale = is_grayscale
        self.use_bicubic_interpolation = use_bicubic_interpolation
        self.pixels = np.zeros((height, width), dtype=np.float32)

    def set_pixel(self, x, y, value):
        self.pixels[y, x] = value

    def get_texture_2d(self):
        return Image.fromarray(np.uint8(self.pixels * 255))

class MathFunctions:
    @staticmethod
    def lat_lon_to_xyz(lat, lon, radius):
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        x = radius * np.cos(lat_rad) * np.cos(lon_rad)
        y = radius * np.cos(lat_rad) * np.sin(lon_rad)
        z = radius * np.sin(lat_rad)
        return np.array([x, y, z])

class PreviewNode(HeightmapNode):
    def __init__(self):
        super().__init__()
        self.input = None
        self.width = 512
        self.height = 256
        self.auto_update_preview = False

    def update_preview(self):
        m = self.get_module()
        self.preview_heightmap = Heightmap(self.width, self.height, False, False)
        x_mul = 360.0 / self.width
        y_mul = 180.0 / self.height

        for x in range(self.width):
            for y in range(self.height):
                lat = y * y_mul
                lon = x * x_mul
                xyz = MathFunctions.lat_lon_to_xyz(lat, lon, 1.0)
                value = max(0, min(1, (m.get_noise(*xyz) + 1) * 0.5))
                self.preview_heightmap.set_pixel(x, y, value)

        self.preview_changed = True

    def get_module(self):
        return ModuleWrapper.zero().module  # Placeholder for getting the module

class ModuleWrapper:
    def __init__(self, module):
        self.module = module

    @staticmethod
    def zero():
        return ModuleWrapper(Module())

class DensityProfileLayer:
    def __init__(self, name, width, exp_term, exp_scale, linear_term, constant_term):
        self.name = name
        self.width = width
        self.exp_term = exp_term
        self.exp_scale = exp_scale
        self.linear_term = linear_term
        self.constant_term = constant_term

# Example usage:
heightmap_node_editor = HeightmapNodeEditor()
heightmap_node_editor.target = HeightmapNode()
heightmap_node_editor.on_body_gui()

preview_node = PreviewNode()
preview_node.update_preview()
print(preview_node.preview_heightmap.get_texture_2d())

density_profile_layer = DensityProfileLayer("Layer1", 1000, 1, 0.001, 0.01, 0.5)

import numpy as np

class Vector2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Vector2({self.x}, {self.y})"

class Matrix2x2:
    def __init__(self, column0=None, column1=None):
        if column0 is None:
            column0 = Vector2(0, 0)
        if column1 is None:
            column1 = Vector2(0, 0)
        self.m00 = column0.x
        self.m10 = column0.y
        self.m01 = column1.x
        self.m11 = column1.y

    def __mul__(self, other):
        if isinstance(other, Vector2):
            return Vector2(
                self.m00 * other.x + self.m01 * other.y,
                self.m10 * other.x + self.m11 * other.y
            )
        elif isinstance(other, (int, float)):
            return Matrix2x2(
                Vector2(self.m00 * other, self.m10 * other),
                Vector2(self.m01 * other, self.m11 * other)
            )
        else:
            raise TypeError("Unsupported multiplication")

    @property
    def inverse(self):
        det = self.determinant
        if det == 0:
            raise ValueError("Matrix is not invertible")
        inv_det = 1.0 / det
        return Matrix2x2(
            Vector2(self.m11 * inv_det, -self.m10 * inv_det),
            Vector2(-self.m01 * inv_det, self.m00 * inv_det)
        )

    @property
    def determinant(self):
        return self.m00 * self.m11 - self.m01 * self.m10

    def __repr__(self):
        return f"Matrix2x2([{self.m00}, {self.m01}], [{self.m10}, {self.m11}])"

class Module:
    def __init__(self):
        pass

class FastNoise(Module):
    def __init__(self, seed):
        super().__init__()
        self.seed = seed
        self.noise_type = None
        self.fractal_type = None
        self.octaves = 1
        self.frequency = 1.0
        self.lacunarity = 2.0

    def set_noise_type(self, noise_type):
        self.noise_type = noise_type

    def set_fractal_type(self, fractal_type):
        self.fractal_type = fractal_type

    def set_fractal_octaves(self, octaves):
        self.octaves = octaves

    def set_frequency(self, frequency):
        self.frequency = frequency

    def set_fractal_lacunarity(self, lacunarity):
        self.lacunarity = lacunarity

class ModuleWrapper:
    def __init__(self, module):
        self.module = module

class PTNode:
    def __init__(self):
        self.output = None

    def get_value(self, port):
        if port != "output":
            return None
        return ModuleWrapper(self.get_module())

    def get_module(self):
        return None

class NoiseType:
    SimplexFractal = 0

class FractalType:
    Billow = 0

class GeneratorNode(PTNode):
    def __init__(self):
        super().__init__()
        self.noise_type = NoiseType.SimplexFractal
        self.fractal_type = FractalType.Billow
        self.seed = 42
        self.octaves = 20
        self.frequency = 1.0
        self.lacunarity = 2.0

    def get_value(self, port):
        if port != "output":
            return None
        return ModuleWrapper(self.get_module())

    def get_module(self):
        noise = FastNoise(self.seed)
        noise.set_noise_type(self.noise_type)
        noise.set_fractal_type(self.fractal_type)
        noise.set_fractal_octaves(self.octaves)
        noise.set_frequency(self.frequency)
        noise.set_fractal_lacunarity(self.lacunarity)
        return noise

# Example usage
column0 = Vector2(1, 2)
column1 = Vector2(3, 4)
matrix = Matrix2x2(column0, column1)
vector = Vector2(5, 6)

print("Matrix:", matrix)
print("Vector:", vector)
print("Matrix * Vector:", matrix * vector)

print("Matrix Inverse:", matrix.inverse)
print("Matrix Determinant:", matrix.determinant)

generator_node = GeneratorNode()
module_wrapper = generator_node.get_value("output")
print("Generated module:", module_wrapper.module)

import threading
import numpy as np
from PIL import Image

class EditorGUI:
    @staticmethod
    def int_field(label, value):
        return value  # Placeholder for integer field

    @staticmethod
    def float_field(label, value):
        return value  # Placeholder for float field

    @staticmethod
    def enum_popup(label, enum_value):
        return enum_value  # Placeholder for enum popup

    @staticmethod
    def color_field(label, color):
        return color  # Placeholder for color field

class GUILayout:
    @staticmethod
    def label(content, style=None):
        print(content)  # Placeholder for GUI label

    @staticmethod
    def button(label):
        print(f"Button: {label}")
        return True  # Always return True for simplicity

class EditorGUILayout:
    @staticmethod
    def object_field(label, obj, obj_type, allow_scene_objects):
        return obj  # Placeholder for object field

class NodeEditorWindow:
    current = None

    @staticmethod
    def repaint():
        print("Repainting window")  # Placeholder for repainting the editor window

class GUIStyle:
    def __init__(self):
        self.alignment = None

class TextAnchor:
    UpperCenter = None

class NodeEditor:
    def on_body_gui(self):
        self.on_body_gui_light()

    def on_body_gui_light(self):
        pass


class NoiseType:
    SimplexFractal = 0
    CubicFractal = 1
    PerlinFractal = 2
    ValueFractal = 3

class FractalType:
    Billow = 0

class ModuleType:
    REMAP = 0

class Module:
    def __init__(self):
        pass

class FastNoise(Module):
    def __init__(self, seed):
        super().__init__()
        self.seed = seed
        self.noise_type = None
        self.fractal_type = None
        self.octaves = 1
        self.frequency = 1.0
        self.lacunarity = 2.0

    def set_noise_type(self, noise_type):
        self.noise_type = noise_type

    def set_fractal_type(self, fractal_type):
        self.fractal_type = fractal_type

    def set_fractal_octaves(self, octaves):
        self.octaves = octaves

    def set_frequency(self, frequency):
        self.frequency = frequency

    def set_fractal_lacunarity(self, lacunarity):
        self.lacunarity = lacunarity

class ModuleWrapper:
    def __init__(self, module):
        self.module = module

class PTNode:
    def __init__(self):
        self.output = None
        self.auto_update_preview = False
        self.preview = None
        self.preview_changed = False
        self.preview_heightmap = None

    def get_value(self, port):
        if port != "output":
            return None
        return ModuleWrapper(self.get_module())

    def get_module(self):
        return None

    def update_preview(self):
        pass  # Placeholder for updating preview

class GeneratorNode(PTNode):
    def __init__(self):
        super().__init__()
        self.noise_type = NoiseType.SimplexFractal
        self.fractal_type = FractalType.Billow
        self.seed = 42
        self.octaves = 20
        self.frequency = 1.0
        self.lacunarity = 2.0

    def get_value(self, port):
        if port != "output":
            return None
        return ModuleWrapper(self.get_module())

    def get_module(self):
        noise = FastNoise(self.seed)
        noise.set_noise_type(self.noise_type)
        noise.set_fractal_type(self.fractal_type)
        noise.set_fractal_octaves(self.octaves)
        noise.set_frequency(self.frequency)
        noise.set_fractal_lacunarity(self.lacunarity)
        return noise

class GeneratorNodeEditor(NodeEditor):
    def on_body_gui(self):
        node = self.target
        super().on_body_gui_light()

        node.seed = EditorGUI.int_field("Seed", node.seed)
        node.frequency = EditorGUI.float_field("Frequency", node.frequency)

        using_fractal_noise = node.noise_type in [
            NoiseType.CubicFractal, NoiseType.PerlinFractal, NoiseType.SimplexFractal, NoiseType.ValueFractal]
        if using_fractal_noise:
            node.octaves = EditorGUI.int_field("Octaves", node.octaves)
            node.lacunarity = EditorGUI.float_field("Lacunarity", node.lacunarity)
        node.noise_type = EditorGUI.enum_popup("Noise Type", node.noise_type)
        if using_fractal_noise:
            node.fractal_type = EditorGUI.enum_popup("Fractal Type", node.fractal_type)

        if node.preview is None:
            threading.Thread(target=node.update_preview).start()

        if node.preview_changed:
            if node.preview_heightmap is None:
                return
            node.preview = node.preview_heightmap.get_texture_2d()
            node.preview_changed = False
            NodeEditorWindow.repaint()

        centered = GUIStyle()
        centered.alignment = TextAnchor.UpperCenter

        GUILayout.label(node.preview, centered)

class Texture2D:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.image = Image.new("RGB", (width, height))

    def get_pixel(self, x, y):
        return self.image.getpixel((x, y))

class Color32:
    def __init__(self, r, g, b, a):
        self.r = r
        self.g = g
        self.b = b
        self.a = a

class Color:
    black = Color32(0, 0, 0, 255)

class EditorStyles:
    boldLabel = None




generator_node_editor = GeneratorNodeEditor()
generator_node_editor.target = GeneratorNode()
generator_node_editor.on_body_gui()


import cupy as cp
import numpy as np

class FastNoise:
    def __init__(self, seed, frequency, octaves, lacunarity, gain, fractal_type, interp, cellular_distance_function, cellular_return_type, cellular_jitter=0.45):
        self.seed = seed
        self.frequency = frequency
        self.octaves = octaves
        self.lacunarity = lacunarity
        self.gain = gain
        self.fractal_type = fractal_type
        self.interp = interp
        self.cellular_distance_function = cellular_distance_function
        self.cellular_return_type = cellular_return_type
        self.cellular_jitter = cellular_jitter
        self.fractal_bounding = self.calculate_fractal_bounding()

    def calculate_fractal_bounding(self):
        amp = self.gain
        ampFractal = 1.0
        for _ in range(1, self.octaves):
            ampFractal += amp
            amp *= self.gain
        return 1.0 / ampFractal

    @staticmethod
    def fast_floor(x):
        return cp.floor(x).astype(int)

    @staticmethod
    def fast_round(x):
        return cp.round(x).astype(int)

    @staticmethod
    def abs(x):
        return cp.abs(x)

    def hash_3d(self, seed, x, y, z):
        n = seed
        n ^= 1619 * x
        n ^= 31337 * y
        n ^= 6971 * z
        n = n * n * n * 60493
        n = (n >> 13) ^ n
        return n

    def val_coord_3d(self, seed, x, y, z):
        n = self.hash_3d(seed, x, y, z)
        return n / 2147483648.0

    def single_cellular(self, x, y, z):
        xr = int(self.fast_round(x))
        yr = int(self.fast_round(y))
        zr = int(self.fast_round(z))

        distance = 999999.0
        xc, yc, zc = 0, 0, 0

        for xi in range(xr - 1, xr + 2):
            for yi in range(yr - 1, yr + 2):
                for zi in range(zr - 1, zr + 2):
                    vec = self.hash_3d(self.seed, xi, yi, zi) & 255
                    vecX = xi - x + vec * self.cellular_jitter
                    vecY = yi - y + vec * self.cellular_jitter
                    vecZ = zi - z + vec * self.cellular_jitter

                    if self.cellular_distance_function == 'Euclidean':
                        new_distance = vecX * vecX + vecY * vecY + vecZ * vecZ
                    elif self.cellular_distance_function == 'Manhattan':
                        new_distance = self.abs(vecX) + self.abs(vecY) + self.abs(vecZ)
                    elif self.cellular_distance_function == 'Natural':
                        new_distance = (self.abs(vecX) + self.abs(vecY) + self.abs(vecZ)) + (vecX * vecX + vecY * vecY + vecZ * vecZ)

                    if new_distance < distance:
                        distance = new_distance
                        xc, yc, zc = xi, yi, zi

        if self.cellular_return_type == 'CellValue':
            return self.val_coord_3d(self.seed, xc, yc, zc)
        elif self.cellular_return_type == 'NoiseLookup':
            vec = self.hash_3d(self.seed, xc, yc, zc) & 255
            return self.get_noise(xc + vec * self.cellular_jitter, yc + vec * self.cellular_jitter, zc + vec * self.cellular_jitter)
        elif self.cellular_return_type == 'Distance':
            return distance
        else:
            return 0

    def single_cellular_2_edge(self, x, y, z):
        xr = int(self.fast_round(x))
        yr = int(self.fast_round(y))
        zr = int(self.fast_round(z))

        distance = cp.array([999999.0, 999999.0, 999999.0, 999999.0])

        for xi in range(xr - 1, xr + 2):
            for yi in range(yr - 1, yr + 2):
                for zi in range(zr - 1, zr + 2):
                    vec = self.hash_3d(self.seed, xi, yi, zi) & 255
                    vecX = xi - x + vec * self.cellular_jitter
                    vecY = yi - y + vec * self.cellular_jitter
                    vecZ = zi - z + vec * self.cellular_jitter

                    if self.cellular_distance_function == 'Euclidean':
                        new_distance = vecX * vecX + vecY * vecY + vecZ * vecZ
                    elif self.cellular_distance_function == 'Manhattan':
                        new_distance = self.abs(vecX) + self.abs(vecY) + self.abs(vecZ)
                    elif self.cellular_distance_function == 'Natural':
                        new_distance = (self.abs(vecX) + self.abs(vecY) + self.abs(vecZ)) + (vecX * vecX + vecY * vecY + vecZ * vecZ)

                    for i in range(3, 0, -1):
                        distance[i] = cp.maximum(cp.minimum(distance[i], new_distance), distance[i - 1])
                    distance[0] = cp.minimum(distance[0], new_distance)

        return distance

    def get_noise(self, x, y, z):
        # Implement the noise function according to your noise type
        pass

# Example usage
noise = FastNoise(seed=42, frequency=0.01, octaves=3, lacunarity=2.0, gain=0.5, fractal_type='FBM', interp='Quintic', cellular_distance_function='Euclidean', cellular_return_type='CellValue')
result = noise.single_cellular(1.0, 2.0, 3.0)
print(result)

import cupy as cp

class FastNoise:
    def __init__(self, seed, frequency, octaves, lacunarity, gain, fractal_type, interp, cellular_distance_function, cellular_return_type, cellular_jitter=0.45, cellular_distance_index0=0, cellular_distance_index1=1, gradient_perturb_amp=1.0):
        self.seed = seed
        self.frequency = frequency
        self.octaves = octaves
        self.lacunarity = lacunarity
        self.gain = gain
        self.fractal_type = fractal_type
        self.interp = interp
        self.cellular_distance_function = cellular_distance_function
        self.cellular_return_type = cellular_return_type
        self.cellular_jitter = cellular_jitter
        self.cellular_distance_index0 = cellular_distance_index0
        self.cellular_distance_index1 = cellular_distance_index1
        self.gradient_perturb_amp = gradient_perturb_amp
        self.fractal_bounding = self.calculate_fractal_bounding()

    def calculate_fractal_bounding(self):
        amp = self.gain
        ampFractal = 1.0
        for _ in range(1, self.octaves):
            ampFractal += amp
            amp *= self.gain
        return 1.0 / ampFractal

    @staticmethod
    def fast_floor(x):
        return cp.floor(x).astype(int)

    @staticmethod
    def fast_round(x):
        return cp.round(x).astype(int)

    @staticmethod
    def abs(x):
        return cp.abs(x)

    def hash_3d(self, seed, x, y, z):
        n = seed
        n ^= 1619 * x
        n ^= 31337 * y
        n ^= 6971 * z
        n = n * n * n * 60493
        n = (n >> 13) ^ n
        return n

    def val_coord_3d(self, seed, x, y, z):
        n = self.hash_3d(seed, x, y, z)
        return n / 2147483648.0

    def single_cellular(self, x, y, z):
        xr = int(self.fast_round(x))
        yr = int(self.fast_round(y))
        zr = int(self.fast_round(z))

        distance = 999999.0
        xc, yc, zc = 0, 0, 0

        for xi in range(xr - 1, xr + 2):
            for yi in range(yr - 1, yr + 2):
                for zi in range(zr - 1, zr + 2):
                    vec = self.hash_3d(self.seed, xi, yi, zi) & 255
                    vecX = xi - x + vec * self.cellular_jitter
                    vecY = yi - y + vec * self.cellular_jitter
                    vecZ = zi - z + vec * self.cellular_jitter

                    if self.cellular_distance_function == 'Euclidean':
                        new_distance = vecX * vecX + vecY * vecY + vecZ * vecZ
                    elif self.cellular_distance_function == 'Manhattan':
                        new_distance = self.abs(vecX) + self.abs(vecY) + self.abs(vecZ)
                    elif self.cellular_distance_function == 'Natural':
                        new_distance = (self.abs(vecX) + self.abs(vecY) + self.abs(vecZ)) + (vecX * vecX + vecY * vecY + vecZ * vecZ)

                    if new_distance < distance:
                        distance = new_distance
                        xc, yc, zc = xi, yi, zi

        if self.cellular_return_type == 'CellValue':
            return self.val_coord_3d(self.seed, xc, yc, zc)
        elif self.cellular_return_type == 'NoiseLookup':
            vec = self.hash_3d(self.seed, xc, yc, zc) & 255
            return self.get_noise(xc + vec * self.cellular_jitter, yc + vec * self.cellular_jitter, zc + vec * self.cellular_jitter)
        elif self.cellular_return_type == 'Distance':
            return distance
        else:
            return 0

    def single_cellular_2_edge(self, x, y, z):
        xr = int(self.fast_round(x))
        yr = int(self.fast_round(y))
        zr = int(self.fast_round(z))

        distance = cp.array([999999.0, 999999.0, 999999.0, 999999.0])

        for xi in range(xr - 1, xr + 2):
            for yi in range(yr - 1, yr + 2):
                for zi in range(zr - 1, zr + 2):
                    vec = self.hash_3d(self.seed, xi, yi, zi) & 255
                    vecX = xi - x + vec * self.cellular_jitter
                    vecY = yi - y + vec * self.cellular_jitter
                    vecZ = zi - z + vec * self.cellular_jitter

                    if self.cellular_distance_function == 'Euclidean':
                        new_distance = vecX * vecX + vecY * vecY + vecZ * vecZ
                    elif self.cellular_distance_function == 'Manhattan':
                        new_distance = self.abs(vecX) + self.abs(vecY) + self.abs(vecZ)
                    elif self.cellular_distance_function == 'Natural':
                        new_distance = (self.abs(vecX) + self.abs(vecY) + self.abs(vecZ)) + (vecX * vecX + vecY * vecY + vecZ * vecZ)

                    for i in range(self.cellular_distance_index1, 0, -1):
                        distance[i] = cp.maximum(cp.minimum(distance[i], new_distance), distance[i - 1])
                    distance[0] = cp.minimum(distance[0], new_distance)

        if self.cellular_return_type == 'Distance2':
            return distance[self.cellular_distance_index1]
        elif self.cellular_return_type == 'Distance2Add':
            return distance[self.cellular_distance_index1] + distance[self.cellular_distance_index0]
        elif self.cellular_return_type == 'Distance2Sub':
            return distance[self.cellular_distance_index1] - distance[self.cellular_distance_index0]
        elif self.cellular_return_type == 'Distance2Mul':
            return distance[self.cellular_distance_index1] * distance[self.cellular_distance_index0]
        elif self.cellular_return_type == 'Distance2Div':
            return distance[self.cellular_distance_index0] / distance[self.cellular_distance_index1]
        else:
            return 0

    def get_cellular(self, x, y):
        x *= self.frequency
        y *= self.frequency

        if self.cellular_return_type in ['CellValue', 'NoiseLookup', 'Distance']:
            return self.single_cellular(x, y)
        else:
            return self.single_cellular_2_edge(x, y)

    def gradient_perturb(self, x, y, z):
        self.single_gradient_perturb(self.seed, self.gradient_perturb_amp, self.frequency, x, y, z)

    def gradient_perturb_fractal(self, x, y, z):
        seed = self.seed
        amp = self.gradient_perturb_amp * self.fractal_bounding
        freq = self.frequency

        self.single_gradient_perturb(seed, amp, freq, x, y, z)

        for i in range(1, self.octaves):
            freq *= self.lacunarity
            amp *= self.gain
            self.single_gradient_perturb(seed + i, amp, freq, x, y, z)

    def single_gradient_perturb(self, seed, perturb_amp, frequency, x, y, z):
        xf = x * frequency
        yf = y * frequency
        zf = z * frequency

        x0 = self.fast_floor(xf)
        y0 = self.fast_floor(yf)
        z0 = self.fast_floor(zf)
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        xs, ys, zs = xf - x0, yf - y0, zf - z0
        if self.interp == 'Hermite':
            xs = self.interp_hermite_func(xs)
            ys = self.interp_hermite_func(ys)
            zs = self.interp_hermite_func(zs)
        elif self.interp == 'Quintic':
            xs = self.interp_quintic_func(xs)
            ys = self.interp_quintic_func(ys)
            zs = self.interp_quintic_func(zs)

        vec0 = self.hash_3d(seed, x0, y0, z0) & 255
        vec1 = self.hash_3d(seed, x1, y0, z0) & 255

        lx0x = self.lerp(vec0, vec1, xs)
        ly0x = self.lerp(vec0, vec1, xs)
        lz0x = self.lerp(vec0, vec1, xs)

        vec0 = self.hash_3d(seed, x0, y1, z0) & 255
        vec1 = self.hash_3d(seed, x1, y1, z0) & 255

        lx1x = self.lerp(vec0, vec1, xs)
        ly1x = self.lerp(vec0, vec1, xs)
        lz1x = self.lerp(vec0, vec1, xs)

        lx0y = self.lerp(lx0x, lx1x, ys)
        ly0y = self.lerp(ly0x, ly1x, ys)
        lz0y = self.lerp(lz0x, lz1x, ys)

        vec0 = self.hash_3d(seed, x0, y0, z1) & 255
        vec1 = self.hash_3d(seed, x1, y0, z1) & 255

        lx0x = self.lerp(vec0, vec1, xs)
        ly0x = self.lerp(vec0, vec1, xs)
        lz0x = self.lerp(vec0, vec1, xs)

        vec0 = self.hash_3d(seed, x0, y1, z1) & 255
        vec1 = self.hash_3d(seed, x1, y1, z1) & 255

        lx1x = self.lerp(vec0, vec1, xs)
        ly1x = self.lerp(vec0, vec1, xs)
        lz1x = self.lerp(vec0, vec1, xs)

        x += self.lerp(lx0y, self.lerp(lx0x, lx1x, ys), zs) * perturb_amp
        y += self.lerp(ly0y, self.lerp(ly0x, ly1x, ys), zs) * perturb_amp
        z += self.lerp(lz0y, self.lerp(lz0x, lz1x, ys), zs) * perturb_amp

    def single_gradient_perturb(self, seed, perturb_amp, frequency, x, y):
        xf = x * frequency
        yf = y * frequency

        x0 = self.fast_floor(xf)
        y0 = self.fast_floor(yf)
        x1 = x0 + 1
        y1 = y0 + 1

        xs, ys = xf - x0, yf - y0
        if self.interp == 'Hermite':
            xs = self.interp_hermite_func(xs)
            ys = self.interp_hermite_func(ys)
        elif self.interp == 'Quintic':
            xs = self.interp_quintic_func(xs)
            ys = self.interp_quintic_func(ys)

        vec0 = self.hash_3d(seed, x0, y0, 0) & 255
        vec1 = self.hash_3d(seed, x1, y0, 0) & 255

        lx0x = self.lerp(vec0, vec1, xs)
        ly0x = self.lerp(vec0, vec1, xs)

        vec0 = self.hash_3d(seed, x0, y1, 0) & 255
        vec1 = self.hash_3d(seed, x1, y1, 0) & 255

        lx1x = self.lerp(vec0, vec1, xs)
        ly1x = self.lerp(vec0, vec1, xs)

        x += self.lerp(lx0x, lx1x, ys) * perturb_amp
        y += self.lerp(ly0x, ly1x, ys) * perturb_amp

    @staticmethod
    def lerp(a, b, t):
        return a + t * (b - a)

    @staticmethod
    def interp_hermite_func(t):
        return t * t * (3 - 2 * t)

    @staticmethod
    def interp_quintic_func(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def get_noise(self, x, y, z):
        # Implement the noise function according to your noise type
        pass

# Example usage
noise = FastNoise(seed=42, frequency=0.01, octaves=3, lacunarity=2.0, gain=0.5, fractal_type='FBM', interp='Quintic', cellular_distance_function='Euclidean', cellular_return_type='CellValue')
result = noise.single_cellular(1.0, 2.0, 3.0)
print(result)



import numpy as np

class Transform:
    def __init__(self, position):
        self.position = np.array(position)

class MathFunctions:
    @staticmethod
    def rotate_around_point(position, point, rotation):
        # Assuming rotation is an array of angles in radians
        # Creating a rotation matrix for each axis
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rotation[0]), -np.sin(rotation[0])],
            [0, np.sin(rotation[0]), np.cos(rotation[0])]
        ])
        Ry = np.array([
            [np.cos(rotation[1]), 0, np.sin(rotation[1])],
            [0, 1, 0],
            [-np.sin(rotation[1]), 0, np.cos(rotation[1])]
        ])
        Rz = np.array([
            [np.cos(rotation[2]), -np.sin(rotation[2]), 0],
            [np.sin(rotation[2]), np.cos(rotation[2]), 0],
            [0, 0, 1]
        ])
        R = Rz @ Ry @ Rx  # Combined rotation matrix
        return R @ (position - point) + point

class RotateAroundPlanet:
    def __init__(self, transform, rotation_speed, parent_planet):
        self.transform = transform
        self.rotation_speed = rotation_speed
        self.parent_planet = parent_planet

    def fixed_update(self):
        rot_speed = np.array([np.radians(angle) for angle in self.rotation_speed])
        self.transform.position = MathFunctions.rotate_around_point(self.transform.position, self.parent_planet.position, rot_speed)

# Example usage
object_transform = Transform([1, 0, 0])
parent_planet_transform = Transform([0, 0, 0])

rotate_around_planet = RotateAroundPlanet(object_transform, [0, 0.000005, 0], parent_planet_transform)
rotate_around_planet.fixed_update()

print(object_transform.position)
import numpy as np

# Placeholder classes for Unity components
class MonoBehaviour:
    def __init__(self):
        self.transform = Transform()

class Transform:
    def __init__(self):
        self.children = []
        self.rotation = QuaternionD.identity()

class Planet:
    def __init__(self):
        self.rotation = QuaternionD.identity()

class Vector3d:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class QuaternionD:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def euler(vector3d):
        cx = np.cos(vector3d.x / 2)
        cy = np.cos(vector3d.y / 2)
        cz = np.cos(vector3d.z / 2)
        sx = np.sin(vector3d.x / 2)
        sy = np.sin(vector3d.y / 2)
        sz = np.sin(vector3d.z / 2)

        w = cx * cy * cz + sx * sy * sz
        x = sx * cy * cz - cx * sy * sz
        y = cx * sy * cz + sx * cy * sz
        z = cx * cy * sz - sx * sy * cz

        return QuaternionD(w, x, y, z)

    @staticmethod
    def identity():
        return QuaternionD(1.0, 0.0, 0.0, 0.0)

    def __mul__(self, other):
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        return QuaternionD(w, x, y, z)

class Rotate(MonoBehaviour):
    def __init__(self):
        super().__init__()
        self.rotation_speed = Vector3d(0.0, 0.0001, 0.0)
        self.r_speed_q = QuaternionD.euler(self.rotation_speed)
        self.planets = []

    def start(self):
        planet = self.get_component(Planet)
        if planet:
            self.planets.append(planet)

        for child in self.transform.children:
            planet = child.get_component(Planet)
            if planet:
                self.planets.append(planet)

        self.r_speed_q = QuaternionD.euler(self.rotation_speed)

    def fixed_update(self):
        self.r_speed_q = QuaternionD.euler(self.rotation_speed)
        for planet in self.planets:
            planet.rotation *= self.r_speed_q

    def get_component(self, component_type):
        return Planet()  # Placeholder for getting a component

# Example usage:
rotate = Rotate()
rotate.start()
rotate.fixed_update()

for planet in rotate.planets:
    print(f'Planet rotation: w={planet.rotation.w}, x={planet.rotation.x}, y={planet.rotation.y}, z={planet.rotation.z}')


# Placeholder classes and methods for Unity Editor components

class EditorWindow:
    @staticmethod
    def show_window():
        pass

class EditorGUILayout:
    @staticmethod
    def object_field(label, obj, obj_type, allow_scene_objects):
        return obj

    @staticmethod
    def text_field(label, text):
        return text

    @staticmethod
    def enum_popup(label, selected_enum):
        return selected_enum

class GUILayout:
    @staticmethod
    def button(label):
        return True

class Texture2D:
    def __init__(self, width, height):
        self.width = width
        self.height = height

class Channel:
    R = 0
    G = 1
    B = 2
    A = 3

class Heightmap:
    def __init__(self, texture, mipmaps, channel):
        self.texture = texture
        self.mipmaps = mipmaps
        self.channel = channel

    def get_file_bytes(self):
        # Return dummy data for example purposes
        return b'\x00' * (self.texture.width * self.texture.height)

# Placeholder classes and methods for Unity Editor components

class EditorWindow:
    @staticmethod
    def show_window():
        pass

class EditorGUILayout:
    @staticmethod
    def object_field(label, obj, obj_type, allow_scene_objects):
        return obj  # Return the same object for simplicity

    @staticmethod
    def text_field(label, text):
        return text

    @staticmethod
    def enum_popup(label, selected_enum):
        return selected_enum

class GUILayout:
    @staticmethod
    def button(label):
        return True

class Texture2D:
    def __init__(self, width, height):
        self.width = width
        self.height = height

class Channel:
    R = 0
    G = 1
    B = 2
    A = 3

class Heightmap:
    def __init__(self, texture, mipmaps, channel):
        self.texture = texture
        self.mipmaps = mipmaps
        self.channel = channel

    def get_file_bytes(self):
        # Return dummy data for example purposes
        return b'\x00' * (self.texture.width * self.texture.height)

class HeightmapToRaw(EditorWindow):
    def __init__(self):
        self.heightmap = None
        self.channel = Channel.R
        self.filename = "heightmap"

    @staticmethod
    def init():
        window = HeightmapToRaw()
        # Set heightmap before calling on_gui
        window.heightmap = Texture2D(512, 512)  # Example texture
        window.on_gui()

    def on_gui(self):
        self.heightmap = EditorGUILayout.object_field("Heightmap", self.heightmap, Texture2D, False)
        self.filename = EditorGUILayout.text_field("Filename", self.filename)
        self.channel = EditorGUILayout.enum_popup("Source Channel", self.channel)

        if GUILayout.button("Convert"):
            heightmap_raw = Heightmap(self.heightmap, False, self.channel)
            print(f"Width: {self.heightmap.width}, Height: {self.heightmap.height}")
            with open(f"{self.filename}.bytes", "wb") as f:
                f.write(heightmap_raw.get_file_bytes())
            # Placeholder for AssetDatabase.Refresh()

# Example usage
heightmap_to_raw = HeightmapToRaw()
heightmap_to_raw.heightmap = Texture2D(512, 512)  # Example texture
heightmap_to_raw.init()

# Implementing PTNodeGraph example
import threading

class PTNode:
    def __init__(self):
        self.outputs = []
        self.auto_update_preview = False

    def update_preview(self):
        print("Updating preview...")

class NodeEditorWindow:
    current = None

class OperatorNode(PTNode):
    def __init__(self):
        super().__init__()
        self.module_type = None
        self.module_type_old = None
        self.inputs = []
        self.parameters = []

class GeneratorNode(PTNode):
    pass

class PTNodeGraph:
    def ripple_update(self, node):
        self.update_node(node)
        for np in node.outputs:
            for connection in np.connections:
                self.ripple_update(connection.node)

    def update_node(self, node):
        if isinstance(node, PTNode):
            if node.auto_update_preview:
                node.update_preview()

    def on_enable(self):
        NodeEditorWindow.current.on_update_node = self.on_update_node

    def on_update_node(self, node):
        if isinstance(node, OperatorNode):
            if node.module_type != node.module_type_old:
                node.module_type_old = node.module_type
                for port in node.inputs:
                    port.clear_connections()
                if node.module_type == ModuleType.REMAP:
                    node.parameters = [1, 1, 1, 0, 0, 0]
        threading.Thread(target=self.ripple_update, args=(node,)).start()

# Example usage
node_graph = PTNodeGraph()
node = GeneratorNode()
node_graph.ripple_update(node)

# Implementing PTNodeGraph example
import threading

class PTNode:
    def __init__(self):
        self.outputs = []
        self.auto_update_preview = False

    def update_preview(self):
        print("Updating preview...")

class NodeEditorWindow:
    current = None

class OperatorNode(PTNode):
    def __init__(self):
        super().__init__()
        self.module_type = None
        self.module_type_old = None
        self.inputs = []
        self.parameters = []

class GeneratorNode(PTNode):
    pass

class PTNodeGraph:
    def ripple_update(self, node):
        self.update_node(node)
        for np in node.outputs:
            for connection in np.connections:
                self.ripple_update(connection.node)

    def update_node(self, node):
        if isinstance(node, PTNode):
            if node.auto_update_preview:
                node.update_preview()

    def on_enable(self):
        NodeEditorWindow.current.on_update_node = self.on_update_node

    def on_update_node(self, node):
        if isinstance(node, OperatorNode):
            if node.module_type != node.module_type_old:
                node.module_type_old = node.module_type
                for port in node.inputs:
                    port.clear_connections()
                if node.module_type == ModuleType.REMAP:
                    node.parameters = [1, 1, 1, 0, 0, 0]
        threading.Thread(target=self.ripple_update, args=(node,)).start()

# Example usage
node_graph = PTNodeGraph()
node = GeneratorNode()
node_graph.ripple_update(node)

# Placeholder classes and methods for Unity Editor components

class EditorWindow:
    @staticmethod
    def show_window():
        pass

class EditorGUILayout:
    @staticmethod
    def object_field(label, obj, obj_type, allow_scene_objects):
        return obj  # Return the same object for simplicity

    @staticmethod
    def color_field(label, color):
        return color

class GUILayout:
    @staticmethod
    def label(text, style):
        print(f"Label: {text}")

    @staticmethod
    def button(label):
        print(f"Button: {label}")
        return True

class EditorStyles:
    boldLabel = None

class Color:
    black = [0, 0, 0]

class Color32:
    def __init__(self, r, g, b, a):
        self.r = r
        self.g = g
        self.b = b
        self.a = a

    def __repr__(self):
        return f"Color32({self.r}, {self.g}, {self.b}, {self.a})"

class Texture2D:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.pixels = [[(i % 256, j % 256, (i*j) % 256) for j in range(height)] for i in range(width)]

    def get_pixel(self, x, y):
        return self.pixels[x][y]

class TextureAverageColor(EditorWindow):
    def __init__(self):
        self.average_color = Color.black
        self.texture = None
        self.r = 0
        self.g = 0
        self.b = 0

    @staticmethod
    def init():
        window = TextureAverageColor()
        window.on_gui()

    def on_gui(self):
        GUILayout.label("Average Texture Color", EditorStyles.boldLabel)
        self.texture = EditorGUILayout.object_field("Texture", self.texture, Texture2D, False)
        self.average_color = EditorGUILayout.color_field("Average Color", self.average_color)

        if GUILayout.button("Calculate"):
            self.average_color = Color.black
            self.r = self.g = self.b = 0
            if self.texture:
                for x in range(self.texture.width):
                    for y in range(self.texture.height):
                        c = self.texture.get_pixel(x, y)
                        self.r += c[0]
                        self.g += c[1]
                        self.b += c[2]
                pixels = self.texture.width * self.texture.height
                self.average_color = Color32(self.r // pixels, self.g // pixels, self.b // pixels, 255)
                print(f"Average Color: {self.average_color}")

# Example usage
texture_average_color = TextureAverageColor()
texture_average_color.texture = Texture2D(512, 512)  # Example texture
texture_average_color.init()

import copy

class Node:
    graph_hotfix = None

    def __init__(self):
        self.graph = None
        self.connections = []

    def clear_connections(self):
        self.connections = []

class NodeGraph:
    def __init__(self):
        self.nodes = []

    def add_node(self, node_class):
        Node.graph_hotfix = self
        node = node_class()
        node.graph = self
        self.nodes.append(node)
        return node

    def copy_node(self, original):
        Node.graph_hotfix = self
        node = copy.deepcopy(original)
        node.graph = self
        node.clear_connections()
        self.nodes.append(node)
        return node

    def remove_node(self, node):
        node.clear_connections()
        self.nodes.remove(node)

    def clear(self):
        self.nodes.clear()

    def copy(self):
        graph_copy = copy.deepcopy(self)
        for node in graph_copy.nodes:
            node.graph = graph_copy
        for node in graph_copy.nodes:
            for port in node.ports:
                port.redirect(self.nodes, graph_copy.nodes)
        return graph_copy

class NodeEditorBase:
    editor_types = {}
    editors = {}

    def __init__(self, target):
        self.target = target
        self.serialized_object = target

    @classmethod
    def get_editor(cls, target):
        if target is None:
            return None
        if target not in cls.editors:
            editor_type = cls.get_editor_type(type(target))
            editor = editor_type(target)
            cls.editors[target] = editor
        editor = cls.editors[target]
        if editor.target is None:
            editor.target = target
        if editor.serialized_object is None:
            editor.serialized_object = target
        return editor

    @classmethod
    def get_editor_type(cls, target_type):
        if target_type is None:
            return None
        if not cls.editor_types:
            cls.cache_custom_editors()
        return cls.editor_types.get(target_type, cls.get_editor_type(target_type.__base__))

    @classmethod
    def cache_custom_editors(cls):
        # This should be populated based on the specific custom editor classes available
        pass

    class INodeEditorAttrib:
        @staticmethod
        def get_inspected_type():
            pass

# Example usage:
class CustomNode(Node):
    def __init__(self):
        super().__init__()
        self.ports = []

class CustomNodeGraph(NodeGraph):
    pass

# Create a new graph and add nodes
graph = CustomNodeGraph()
node = graph.add_node(CustomNode)
copied_node = graph.copy_node(node)
graph.remove_node(node)
graph.clear()

# Copy the entire graph
graph_copy = graph.copy()


import numpy as np

class Vector3d:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vector3d(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3d(self.x - other.x, self.y - other.y, self.z - other.z)

    def __truediv__(self, value):
        return Vector3d(self.x / value, self.y / value, self.z / value)

    def to_tuple(self):
        return (self.x, self.y, self.z)

class Transform:
    def __init__(self, position):
        self.position = position

class Planet:
    def __init__(self):
        self.initialized = False
        self.in_scaled_space = False
        self.quads = []

class FloatingOrigin:
    def __init__(self, threshold=6000.0, player=None, objects=None):
        self.threshold = threshold
        self.player = player
        self.objects = objects if objects is not None else []
        self.distance_from_original_origin = Vector3d()
        self.planets = []

        planet = Planet()
        if planet not in self.planets:
            self.planets.append(planet)

    def start(self):
        if self not in self.planets:
            self.planets.append(self)

    def update(self):
        if (np.linalg.norm(self.player.position) > self.threshold) and (self.planets[0].initialized or self.planets[0].in_scaled_space):
            self.distance_from_original_origin += Vector3d(*self.player.position)
            self.player.position = [self.player.position[i] - self.player.position[i] for i in range(3)]

            for obj in self.objects:
                obj.position = [obj.position[i] - self.player.position[i] for i in range(3)]

            for planet in self.planets:
                for quad in planet.quads:
                    if quad.rendered_quad:
                        quad.rendered_quad.position = [quad.rendered_quad.position[i] - self.player.position[i] for i in range(3)]

            self.player.position = [0, 0, 0]

    def world_space_to_scaled_space(self, world_pos, scale_factor):
        world_pos = Vector3d(*world_pos)
        scaled_pos = (world_pos + self.distance_from_original_origin) / scale_factor
        return scaled_pos.to_tuple()

class FloatCurve:
    def __init__(self, times=None, values=None):
        self.times = times if times is not None else [0.0, 0.25, 0.75, 1.0]
        self.values = values if values is not None else [0.0, 0.0625, 0.5625, 1.0]

    def evaluate(self, time):
        if len(self.times) == len(self.values) and len(self.times) > 0:
            time = (time + 1.0) / 2.0
            index = next((i for i, t in enumerate(self.times) if time < t), len(self.times) - 1)

            length = len(self.times) - 1
            index0 = max(index - 2, 0)
            index1 = max(index - 1, 0)
            index2 = min(index, length)
            index3 = min(index + 1, length)

            if index1 == index2:
                return self.values[index1]

            alpha = (time - self.times[index1]) / (self.times[index2] - self.times[index1])
            return 2.0 * self.cubic_interpolation(self.values[index0], self.values[index1], self.values[index2], self.values[index3], alpha) - 1.0
        return 0.0

    @staticmethod
    def cubic_interpolation(v0, v1, v2, v3, t):
        return ((v3 - v2 - v0 + v1) * t ** 3 + (v0 - v1 - v3 + v2) * t ** 2 + (v2 - v0) * t + v1)

# Example usage:
player_transform = Transform([0, 0, 0])
object_transforms = [Transform([10, 10, 10]), Transform([20, 20, 20])]

floating_origin = FloatingOrigin(player=player_transform, objects=object_transforms)
floating_origin.update()

float_curve = FloatCurve()
print(float_curve.evaluate(0.5))



import numpy as np
from scipy.interpolate import interp1d
from threading import Thread

# A placeholder for the Vector3 structure
class Vector3:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Vector3({self.x}, {self.y}, {self.z})"

# Placeholder for the ComputeBuffer
class ComputeBuffer:
    def __init__(self, length, stride):
        self.data = np.zeros((length, stride // 4))  # Assuming stride is in bytes, and we're using floats

    def set_data(self, data):
        self.data = data

    def dispose(self):
        del self.data

# Placeholder for the ComputeShader
class ComputeShader:
    def find_kernel(self, name):
        return name

    def set_float(self, name, value):
        setattr(self, name, value)

    def set_floats(self, name, values):
        setattr(self, name, values)

    def set_buffer(self, kernel, name, buffer):
        setattr(self, f"{kernel}_{name}", buffer)

    def dispatch(self, kernel, x, y, z):
        pass

class AsyncGPUReadbackRequest:
    def __init__(self, buffer):
        self.done = False
        self.has_error = False
        self.buffer = buffer

    def get_data(self):
        return self.buffer.data

    @staticmethod
    def request(buffer):
        request = AsyncGPUReadbackRequest(buffer)
        Thread(target=request._simulate_gpu_readback).start()
        return request

    def _simulate_gpu_readback(self):
        import time
        time.sleep(0.1)  # Simulate some GPU work
        self.done = True

class CBRead:
    read_names_2d = [
        ["read2DC1", "_Tex2D", "_Buffer2DC1"],
        ["read2DC2", "_Tex2D", "_Buffer2DC2"],
        ["read2DC3", "_Tex2D", "_Buffer2DC3"],
        ["read2DC4", "_Tex2D", "_Buffer2DC4"],
    ]

    read_names_3d = [
        ["read3DC1", "_Tex3D", "_Buffer3DC1"],
        ["read3DC2", "_Tex3D", "_Buffer3DC2"],
        ["read3DC3", "_Tex3D", "_Buffer3DC3"],
        ["read3DC4", "_Tex3D", "_Buffer3DC4"],
    ]

    @staticmethod
    def from_render_texture(tex, channels, buffer, read):
        CBRead.check(tex, channels, buffer, read)

        kernel = -1
        depth = 1

        if tex.dimension == "3D":
            depth = tex.volume_depth
            kernel = read.find_kernel(CBRead.read_names_3d[channels - 1][0])
            read.set_texture(kernel, CBRead.read_names_3d[channels - 1][1], tex)
            read.set_buffer(kernel, CBRead.read_names_3d[channels - 1][2], buffer)
        else:
            kernel = read.find_kernel(CBRead.read_names_2d[channels - 1][0])
            read.set_texture(kernel, CBRead.read_names_2d[channels - 1][1], tex)
            read.set_buffer(kernel, CBRead.read_names_2d[channels - 1][2], buffer)

        if kernel == -1:
            raise ValueError(f"Could not find kernel {CBRead.read_names_2d[channels - 1][0]}")

        width = tex.width
        height = tex.height

        read.set_int("_Width", width)
        read.set_int("_Height", height)
        read.set_int("_Depth", depth)

        pad_x = 0 if width % 8 == 0 else 1
        pad_y = 0 if height % 8 == 0 else 1
        pad_z = 0 if depth % 8 == 0 else 1

        read.dispatch(kernel, max(1, width // 8 + pad_x), max(1, height // 8 + pad_y), max(1, depth // 8 + pad_z))

    @staticmethod
    def single_from_render_texture(tex, x, y, z, buffer, read, use_bilinear):
        CBRead.check(tex, 0, buffer, read)

        kernel = -1
        depth = 1

        if tex.dimension == "3D":
            if use_bilinear:
                kernel = read.find_kernel("readSingleBilinear3D")
            else:
                kernel = read.find_kernel("readSingle3D")

            depth = tex.volume_depth
            read.set_texture(kernel, "_Tex3D", tex)
            read.set_buffer(kernel, "_BufferSingle3D", buffer)
        else:
            if use_bilinear:
                kernel = read.find_kernel("readSingleBilinear2D")
            else:
                kernel = read.find_kernel("readSingle2D")

            read.set_texture(kernel, "_Tex2D", tex)
            read.set_buffer(kernel, "_BufferSingle2D", buffer)

        if kernel == -1:
            raise ValueError(f"Could not find kernel readSingle for {tex.dimension}")

        width = tex.width
        height = tex.height

        read.set_int("_IdxX", int(x))
        read.set_int("_IdxY", int(y))
        read.set_int("_IdxZ", int(z))
        read.set_vector("_UV", [x / (width - 1), y / (height - 1), z / (depth - 1), 0.0])

        read.dispatch(kernel, 1, 1, 1)

    @staticmethod
    def check(tex, channels, buffer, read):
        if tex is None:
            raise ValueError("RenderTexture is null")

        if buffer is None:
            raise ValueError("Buffer is null")

        if read is None:
            raise ValueError("Compute shader is null")

        if channels < 1 or channels > 4:
            raise ValueError("Channels must be 1, 2, 3, or 4")

        if not tex.is_created():
            raise ValueError("Tex has not been created (Call Create() on tex)")

class MeshGenerator:
    def __init__(self, planet, quad):
        self.planet = planet
        self.quad = quad
        self.is_running = False

    def start_generation(self):
        raise NotImplementedError

    def apply_to_mesh(self, mesh):
        raise NotImplementedError

    def dispose(self):
        pass

    @property
    def is_completed(self):
        raise NotImplementedError

class CPUMeshGenerator(MeshGenerator):
    def __init__(self, planet, quad):
        super().__init__(planet, quad)
        self.method = None
        self.cookie = None

    @property
    def is_completed(self):
        return self.cookie is not None and self.cookie.done()

    def start_generation(self):
        def generate():
            md = MeshData(self.planet.quad_arrays.get_extended_plane(), self.planet.quad_size * self.planet.quad_size)
            self.method = self.generate_mesh
            self.cookie = self.method(self.quad, md)

        Thread(target=generate).start()
        self.is_running = True

    def apply_to_mesh(self, mesh):
        result = self.cookie.result()
        self.is_running = False

        mesh.vertices = result.vertices
        mesh.colors32 = result.colors
        mesh.uv = result.uv
        mesh.uv4 = result.uv2
        mesh.normals = result.normals

        self.cookie = None
        self.method = None

    def generate_mesh(self, quad, md):
        # Placeholder for mesh generation logic
        return md

class GPUMeshGenerator(MeshGenerator):
    def __init__(self, planet, quad):
        super().__init__(planet, quad)
        self.method = None
        self.cookie = None
        self.gpu_readback_req = None
        self.compute_buffer = None
        self.is_running_on_gpu = False

    @property
    def is_completed(self):
        if self.is_running_on_gpu and self.gpu_readback_req.done:
            if self.gpu_readback_req.has_error:
                self.compute_buffer.dispose()
                self.compute_buffer = None
                self.start_generation()
            else:
                data = self.gpu_readback_req.get_data()
                md = MeshData(data, self.planet.quad_size * self.planet.quad_size)
                self.method = self.generate_mesh_gpu
                self.cookie = self.method(self.quad, md)
                self.compute_buffer.dispose()
                self.compute_buffer = None
                self.is_running_on_gpu = False

        return self.cookie is not None and self.cookie.done()

    def start_generation(self):
        def generate():
            md = MeshData(self.planet.quad_arrays.get_extended_plane(), self.planet.quad_size * self.planet.quad_size)
            kernel_index = self.planet.compute_shader.find_kernel("ComputePositions")

            self.compute_buffer = ComputeBuffer(md.vertices.shape[0], 12)
            self.compute_buffer.set_data(self.planet.quad_arrays.get_extended_plane())

            self.planet.compute_shader.set_float("scale", self.quad.scale)
            self.planet.compute_shader.set_floats("trPosition", [self.quad.tr_position.x, self.quad.tr_position.y, self.quad.tr_position.z])
            self.planet.compute_shader.set_float("radius", self.planet.radius)
            self.planet.compute_shader.set_floats("rotation", [self.quad.rotation.x, self.quad.rotation.y, self.quad.rotation.z, self.quad.rotation.w])
            self.planet.compute_shader.set_float("noiseDiv", 1 / self.planet.height_scale)
            self.planet.compute_shader.set_buffer(kernel_index, "dataBuffer", self.compute_buffer)

            self.planet.compute_shader.dispatch(kernel_index, int(np.ceil(md.vertices.shape[0] / 256)), 1, 1)

            self.gpu_readback_req = AsyncGPUReadbackRequest.request(self.compute_buffer)
            self.is_running = True
            self.is_running_on_gpu = True

        Thread(target=generate).start()

    def apply_to_mesh(self, mesh):
        result = self.cookie.result()
        self.is_running = False

        mesh.vertices = result.vertices
        mesh.colors32 = result.colors
        mesh.uv = result.uv
        mesh.uv4 = result.uv2
        mesh.normals = result.normals

        self.cookie = None
        self.method = None

    def dispose(self):
        if self.compute_buffer:
            self.compute_buffer.dispose()

class MeshData:
    def __init__(self, vertices, size):
        self.vertices = vertices
        self.colors = np.zeros((size, 4), dtype=np.uint8)
        self.uv = np.zeros((size, 2), dtype=np.float32)
        self.uv2 = np.zeros((size, 2), dtype=np.float32)
        self.normals = np.zeros((size, 3), dtype=np.float32)

# Placeholder for the Planet class
class Planet:
    def __init__(self, quad_size, compute_shader, quad_arrays, radius, height_scale):
        self.quad_size = quad_size
        self.compute_shader = compute_shader
        self.quad_arrays = quad_arrays
        self.radius = radius
        self.height_scale = height_scale

# Placeholder for the Quad class
class Quad:
    def __init__(self, scale, tr_position, rotation):
        self.scale = scale
        self.tr_position = tr_position
        self.rotation = rotation

import numpy as np
from threading import Thread
import json

class NodeEditorResources:
    _dot = None
    _dot_outer = None
    _node_body = None
    _node_highlight = None
    _styles = None

    @staticmethod
    def load_texture(path):
        # Placeholder function for loading textures
        return np.zeros((64, 64, 3))

    @staticmethod
    def dot():
        if NodeEditorResources._dot is None:
            NodeEditorResources._dot = NodeEditorResources.load_texture("xnode_dot")
        return NodeEditorResources._dot

    @staticmethod
    def dot_outer():
        if NodeEditorResources._dot_outer is None:
            NodeEditorResources._dot_outer = NodeEditorResources.load_texture("xnode_dot_outer")
        return NodeEditorResources._dot_outer

    @staticmethod
    def node_body():
        if NodeEditorResources._node_body is None:
            NodeEditorResources._node_body = NodeEditorResources.load_texture("xnode_node")
        return NodeEditorResources._node_body

    @staticmethod
    def node_highlight():
        if NodeEditorResources._node_highlight is None:
            NodeEditorResources._node_highlight = NodeEditorResources.load_texture("xnode_node_highlight")
        return NodeEditorResources._node_highlight

    @staticmethod
    def styles():
        if NodeEditorResources._styles is None:
            NodeEditorResources._styles = NodeEditorResources.Styles()
        return NodeEditorResources._styles

    class Styles:
        def __init__(self):
            self.input_port = {'alignment': 'upper_left', 'padding_left': 10, 'fixed_height': 18}
            self.node_header = {'alignment': 'middle_center', 'font_style': 'bold', 'normal_text_color': 'white'}
            self.node_body = {'normal_background': NodeEditorResources.node_body(), 'border': (32, 32, 32, 32), 'padding': (16, 16, 4, 16)}
            self.node_highlight = {'normal_background': NodeEditorResources.node_highlight(), 'border': (32, 32, 32, 32)}
            self.tooltip = {'alignment': 'middle_center'}

    @staticmethod
    def generate_grid_texture(line, bg):
        tex = np.zeros((64, 64, 3), dtype=np.uint8)
        for y in range(64):
            for x in range(64):
                col = bg
                if y % 16 == 0 or x % 16 == 0:
                    col = tuple(int(c * 0.65) for c in line)
                if y == 63 or x == 63:
                    col = tuple(int(c * 0.35) for c in line)
                tex[y, x] = col
        return tex

    @staticmethod
    def generate_cross_texture(line):
        tex = np.zeros((64, 64, 3), dtype=np.uint8)
        for y in range(64):
            for x in range(64):
                col = line if y == 31 or x == 31 else (0, 0, 0, 0)
                tex[y, x] = col
        return tex

class IHeightProvider:
    def height_at_xyz(self, pos):
        raise NotImplementedError

    def init(self):
        raise NotImplementedError

class HeightmapHeightProvider(IHeightProvider):
    def __init__(self, heightmap_text_asset, use_bicubic_interpolation):
        self.heightmap_text_asset = heightmap_text_asset
        self.use_bicubic_interpolation = use_bicubic_interpolation
        self.heightmap = None

    def height_at_xyz(self, pos):
        return self.heightmap.get_pos_interpolated(pos)

    def init(self):
        self.heightmap = Heightmap(self.heightmap_text_asset, self.use_bicubic_interpolation)

class NoiseHeightProvider(IHeightProvider):
    def __init__(self, noise_serialized):
        self.noise_serialized = noise_serialized
        self.noise = None

    def height_at_xyz(self, pos):
        return (self.noise.get_noise(pos.x, pos.y, pos.z) + 1) * 0.5

    def init(self):
        self.noise = Utils.deserialize_text_asset(self.noise_serialized)
        Utils.randomize_noise(self.noise)

class HybridHeightProvider(IHeightProvider):
    def __init__(self, heightmap_text_asset, noise_serialized, use_bicubic_interpolation, hybrid_mode_noise_div):
        self.heightmap_text_asset = heightmap_text_asset
        self.noise_serialized = noise_serialized
        self.use_bicubic_interpolation = use_bicubic_interpolation
        self.hybrid_mode_noise_div = hybrid_mode_noise_div
        self.heightmap = None
        self.noise = None

    def height_at_xyz(self, pos):
        return self.heightmap.get_pos_interpolated(pos) * (self.hybrid_mode_noise_div - ((self.noise.get_noise(pos.x, pos.y, pos.z) + 1) / 2)) / self.hybrid_mode_noise_div

    def init(self):
        self.noise = Utils.deserialize_text_asset(self.noise_serialized)
        self.heightmap = Heightmap(self.heightmap_text_asset, self.use_bicubic_interpolation)

class StreamingHeightmapHeightProvider(IHeightProvider):
    def __init__(self, heightmap_path, base_heightmap_text_asset, use_bicubic_interpolation, load_size, reload_threshold):
        self.heightmap_path = heightmap_path
        self.base_heightmap_text_asset = base_heightmap_text_asset
        self.use_bicubic_interpolation = use_bicubic_interpolation
        self.load_size = load_size
        self.reload_threshold = reload_threshold
        self.s_heightmap = None
        self.last_position = np.ones(3) * float('inf')
        self.currently_reloading = False

    def height_at_xyz(self, pos):
        return self.s_heightmap.get_pos_interpolated(pos)

    def init(self):
        self.s_heightmap = StreamingHeightmap(self.base_heightmap_text_asset, self.heightmap_path, self.use_bicubic_interpolation)

    def update(self, queue, position):
        if np.linalg.norm(position - self.last_position) > self.reload_threshold and not self.currently_reloading:
            if not queue.is_any_currently_splitting:
                queue.stop = True
                self.currently_reloading = True
                def reload():
                    self.s_heightmap.clear_memory()
                    self.s_heightmap.load_area_into_memory(Utils.xyz_to_uv(position), self.load_size)
                    self.currently_reloading = False
                    queue.stop = False
                Thread(target=reload).start()
                self.last_position = position
            else:
                queue.stop = True

class ConstHeightProvider(IHeightProvider):
    def __init__(self, constant):
        self.constant = constant

    def height_at_xyz(self, pos):
        return self.constant

    def init(self):
        pass

class Utils:
    @staticmethod
    def deserialize_text_asset(text_asset):
        # Placeholder for deserializing text assets
        return json.loads(text_asset)

    @staticmethod
    def randomize_noise(noise):
        # Placeholder for randomizing noise
        pass

    @staticmethod
    def xyz_to_uv(position):
        # Placeholder for converting XYZ to UV
        return position / np.linalg.norm(position)

# Placeholder classes
class Heightmap:
    def __init__(self, text_asset, use_bicubic_interpolation):
        self.data = json.loads(text_asset)
        self.use_bicubic_interpolation = use_bicubic_interpolation

    def get_pos_interpolated(self, pos):
        # Placeholder for interpolating heightmap data
        return np.interp(pos, [0, 1], [0, 1])

class StreamingHeightmap:
    def __init__(self, base_text_asset, path, use_bicubic_interpolation):
        self.data = json.loads(base_text_asset)
        self.path = path
        self.use_bicubic_interpolation = use_bicubic_interpolation

    def clear_memory(self):
        self.data = None

    def load_area_into_memory(self, uv, size):
        # Placeholder for loading area into memory
        pass

    def get_pos_interpolated(self, pos):
        # Placeholder for interpolating heightmap data
        return np.interp(pos, [0, 1], [0, 1])

# Placeholder class for QuadSplitQueue
class QuadSplitQueue:
    def __init__(self):
        self.is_any_currently_splitting = False
        self.stop = False

import numpy as np
import json
from scipy.interpolate import interp1d

class Module:
    pass

class ModuleWrapper:
    Zero = Module()

class ModuleType:
    Const = "Const"
    Select = "Select"
    Curve = "Curve"
    Blend = "Blend"
    Remap = "Remap"
    Add = "Add"
    Subtract = "Subtract"
    Multiply = "Multiply"
    Min = "Min"
    Max = "Max"
    Scale = "Scale"
    ScaleBias = "ScaleBias"
    Abs = "Abs"
    Invert = "Invert"
    Clamp = "Clamp"
    Terrace = "Terrace"

class PTNode:
    def get_input_value(self, name, default):
        # Placeholder for getting input value
        return default

    def get_module(self):
        raise NotImplementedError

    def get_value(self, port):
        raise NotImplementedError

class OperatorNode(PTNode):
    def __init__(self):
        self.output = None
        self.input0 = ModuleWrapper.Zero
        self.input1 = ModuleWrapper.Zero
        self.input2 = ModuleWrapper.Zero
        self.module_type = ModuleType.Const
        self.parameters = None
        self.curve = None
        self.module_type_old = ModuleType.Const

    def get_module(self):
        a = self.get_input_value("input0", ModuleWrapper.Zero).m
        b = self.get_input_value("input1", ModuleWrapper.Zero).m
        c = self.get_input_value("input2", ModuleWrapper.Zero).m

        m = None

        if self.parameters is None:
            self.parameters = [0] * 6
        if self.curve is None:
            self.curve = []

        if self.module_type == ModuleType.Select:
            if a is None or b is None or c is None:
                return None
            m = Select(a, b, c, self.parameters[0], self.parameters[1], self.parameters[2])
        elif self.module_type == ModuleType.Curve:
            if a is None:
                return None
            m = Curve(a, self.curve)
        elif self.module_type == ModuleType.Blend:
            if a is None or b is None:
                return None
            m = Blend(a, b, self.parameters[0])
        elif self.module_type == ModuleType.Remap:
            if len(self.parameters) != 6:
                self.parameters = [0] * 6
            if a is None:
                return None
            m = Remap(a, self.parameters)
        elif self.module_type == ModuleType.Add:
            if a is None or b is None:
                return None
            m = Add(a, b)
        elif self.module_type == ModuleType.Subtract:
            if a is None or b is None:
                return None
            m = Subtract(a, b)
        elif self.module_type == ModuleType.Multiply:
            if a is None or b is None:
                return None
            m = Multiply(a, b)
        elif self.module_type == ModuleType.Min:
            if a is None or b is None:
                return None
            m = Min(a, b)
        elif self.module_type == ModuleType.Max:
            if a is None or b is None:
                return None
            m = Max(a, b)
        elif self.module_type == ModuleType.Scale:
            if a is None:
                return None
            m = Scale(a, self.parameters[0])
        elif self.module_type == ModuleType.ScaleBias:
            if a is None:
                return None
            m = ScaleBias(a, self.parameters[0], self.parameters[1])
        elif self.module_type == ModuleType.Abs:
            if a is None:
                return None
            m = Abs(a)
        elif self.module_type == ModuleType.Invert:
            if a is None:
                return None
            m = Invert(a)
        elif self.module_type == ModuleType.Clamp:
            if a is None:
                return None
            m = Clamp(a, self.parameters[0], self.parameters[1])
        elif self.module_type == ModuleType.Terrace:
            if a is None:
                return None
            m = Terrace(a, False, self.parameters)
        else:
            m = Const(self.parameters[0])

        return m

    def get_value(self, port):
        m = self.get_module()
        if m is not None:
            return ModuleWrapper(m)
        return None

# Placeholder classes for the various module types
class Select(Module):
    def __init__(self, a, b, c, p0, p1, p2):
        pass

class Curve(Module):
    def __init__(self, a, curve):
        pass

class Blend(Module):
    def __init__(self, a, b, param):
        pass

class Remap(Module):
    def __init__(self, a, params):
        pass

class Add(Module):
    def __init__(self, a, b):
        pass

class Subtract(Module):
    def __init__(self, a, b):
        pass

class Multiply(Module):
    def __init__(self, a, b):
        pass

class Min(Module):
    def __init__(self, a, b):
        pass

class Max(Module):
    def __init__(self, a, b):
        pass

class Scale(Module):
    def __init__(self, a, param):
        pass

class ScaleBias(Module):
    def __init__(self, a, p0, p1):
        pass

class Abs(Module):
    def __init__(self, a):
        pass

class Invert(Module):
    def __init__(self, a):
        pass

class Clamp(Module):
    def __init__(self, a, p0, p1):
        pass

class Terrace(Module):
    def __init__(self, a, flag, params):
        pass

class Const(Module):
    def __init__(self, param):
        pass

class OldToNewHeightmap:
    def __init__(self):
        self.old_heightmap = None
        self.filename_new_heightmap = ""
        self.width = 8192
        self.height = 4096
        self.is_16bit = False

    def on_gui(self):
        print("Old To New Heightmap")
        self.filename_new_heightmap = input("Filename for new Heightmap: ")
        self.old_heightmap = input("Heightmap path: ")

        self.width = int(input("Width: "))
        self.height = int(input("Height: "))
        self.is_16bit = bool(input("16bit (True/False): "))

        if input("Convert (Yes/No): ").lower() == "yes":
            with open(self.old_heightmap, "rb") as f:
                old_heightmap_bytes = f.read()

            old_heightmap_bytes_len = len(old_heightmap_bytes)
            self.test_heightmap_resolution_old(old_heightmap_bytes_len, self.width, self.height, self.is_16bit)

            new_heightmap_header = bytearray(9)
            new_heightmap_header[:4] = self.width.to_bytes(4, 'little')
            new_heightmap_header[4:8] = self.height.to_bytes(4, 'little')
            new_heightmap_header[8] = int(self.is_16bit).to_bytes(1, 'little')[0]

            new_heightmap = new_heightmap_header + bytearray(old_heightmap_bytes_len + 9)

            if self.is_16bit:
                old_heightmap_bytes_len_h = old_heightmap_bytes_len // 2
                for i in range(old_heightmap_bytes_len_h):
                    new_heightmap[2 * i + 9] = old_heightmap_bytes[i]
                    new_heightmap[2 * i + 10] = old_heightmap_bytes[i + old_heightmap_bytes_len_h]
            else:
                for i in range(old_heightmap_bytes_len):
                    new_heightmap[i + 9] = old_heightmap_bytes[i]

            with open(self.filename_new_heightmap + ".bytes", "wb") as f:
                f.write(new_heightmap)
            print("Conversion complete.")

    @staticmethod
    def test_heightmap_resolution_old(length, width, height, is_16bit):
        if (length // 2 if is_16bit else length) == height * width:
            return

        if (length - 9) // 2 if is_16bit else length - 9 == height * width:
            raise ValueError("Heightmap was already converted to new format! No need to convert it again.")

        raise ValueError("Heightmap resolution incorrect! Cannot read heightmap!")

# Example usage
old_to_new_heightmap = OldToNewHeightmap()
old_to_new_heightmap.on_gui()

import math

class QuadNeighbor:
    lengthMask = 63

    @staticmethod
    def pow(base, exponent):
        result = 1
        for _ in range(exponent):
            result *= base
        return result

    @staticmethod
    def encode(array):
        index = 0
        mul = 1
        for num in array:
            index += num * mul
            mul *= 4
        size = len(array)
        index = (index << 6) | size
        return index

    @staticmethod
    def decode(index):
        size = int(index & QuadNeighbor.lengthMask)
        index = index >> 6
        index_out = []
        div = 1
        for _ in range(size):
            index_out.append(int((index // div) % 4))
            div *= 4
        return index_out

    @staticmethod
    def append(index, num):
        size = int(index & QuadNeighbor.lengthMask)
        index = index >> 6
        pow = 1
        for _ in range(size):
            pow *= 4
        size += 1
        index += pow * num
        index = (index << 6) | size
        return index

    @staticmethod
    def slice(index):
        size = int(index & QuadNeighbor.lengthMask) - 1
        index = index >> 6
        pow = 1
        for _ in range(size):
            pow *= 4
        index -= pow * ((index // pow) % 4)
        index = (index << 6) | size
        return index

    dict = {
        (0, 0): (1, -1),
        (0, 1): (0, 0),
        (0, 2): (3, -1),
        (0, 3): (2, 0),
        (1, 0): (1, 1),
        (1, 1): (0, -1),
        (1, 2): (3, 1),
        (1, 3): (2, -1),
        (2, 0): (2, -1),
        (2, 1): (3, -1),
        (2, 2): (0, 2),
        (2, 3): (1, 2),
        (3, 0): (2, 3),
        (3, 1): (3, 3),
        (3, 2): (0, -1),
        (3, 3): (1, -1),
    }

    quadEdgeNeighbors = {
        (3, 1, 2): (0, 1, 0, 3),
        (0, 0, 1): (1, 2, 3, 0),
        (3, 0, 2): (0, 1, 1, 2),
        (1, 0, 1): (0, 2, 2, 1),
        (3, 1, 3): (0, 1, 0, 1),
        (3, 0, 1): (1, 3, 1, 0),
        (2, 0, 2): (2, 1, 3, 0),
        (1, 2, 1): (0, 2, 0, 3),
        (2, 1, 2): (2, 1, 2, 1),
        (0, 2, 1): (1, 2, 1, 2),
        (2, 1, 3): (2, 1, 0, 0),
        (2, 2, 1): (1, 3, 0, 0),
    }

    @staticmethod
    def get_neighbor(quad_id, dir):
        quad_id_a = QuadNeighbor.decode(quad_id)
        if (dir, quad_id_a[0], quad_id_a[1]) in QuadNeighbor.quadEdgeNeighbors:
            if QuadNeighbor.at_quad_edge(quad_id_a, dir):
                return QuadNeighbor.encode(QuadNeighbor.quad_edges(quad_id_a, dir))
        quad = quad_id_a[-1]
        for i in reversed(range(len(quad_id_a))):
            result = QuadNeighbor.dict[(dir, quad)]
            quad_id_a[i] = result[0]
            dir = result[1]
            if dir == -1 or i - 1 < 0:
                break
            quad = quad_id_a[i - 1]
        return QuadNeighbor.encode(quad_id_a)

    @staticmethod
    def at_quad_edge(quad_id, dir):
        if dir == 0:
            return not QuadNeighbor.contains(quad_id, 0, 2)
        if dir == 1:
            return not QuadNeighbor.contains(quad_id, 1, 3)
        if dir == 2:
            return not QuadNeighbor.contains(quad_id, 0, 1)
        if dir == 3:
            return not QuadNeighbor.contains(quad_id, 2, 3)
        return False

    @staticmethod
    def contains(quad_id, a, b):
        for q in quad_id[2:]:
            if q == a or q == b:
                return True
        return False

    @staticmethod
    def quad_edges(quad_id_a, dir):
        result = QuadNeighbor.quadEdgeNeighbors[(dir, quad_id_a[0], quad_id_a[1])]
        invert = (quad_id_a[0] == 1 and quad_id_a[1] == 3) or (quad_id_a[0] == 0 and quad_id_a[1] == 1 and dir == 3)
        quad_id_a[0] = result[0]
        quad_id_a[1] = result[1]
        for i in range(2, len(quad_id_a)):
            if invert:
                if quad_id_a[i] == result[2]:
                    quad_id_a[i] = result[3]
                elif quad_id_a[i] == result[3]:
                    quad_id_a[i] = result[2]
            else:
                if quad_id_a[i] == result[2]:
                    quad_id_a[i] = result[3]
        return quad_id_a

    class int4:
        def __init__(self, x, y, z, w):
            self.x = x
            self.y = y
            self.z = z
            self.w = w

    class int3:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

    class int2:
        def __init__(self, x, y):
            self.x = x
            self.y = y


import math
import numpy as np

class MathFunctions:
    DegToRad = math.pi / 180.0
    Rad2Deg = 180.0 / math.pi
    HalfPI = math.pi / 2.0
    PIInv = 1.0 / math.pi
    TwoPIInv = 1.0 / (2.0 * math.pi)
    PIInvF = 1.0 / math.pi
    TwoPIInvF = 1.0 / (2.0 * math.pi)
    HalfPIf = math.pi / 2.0
    TwoPIf = math.pi * 2.0

    @staticmethod
    def lat_lon_to_xyz(lat, lon, radius):
        lat *= MathFunctions.DegToRad
        lon *= MathFunctions.DegToRad
        x = -radius * math.sin(lat) * math.cos(lon)
        y = -radius * math.cos(lat)
        z = -radius * math.sin(lat) * math.sin(lon)
        return np.array([x, y, z])

    @staticmethod
    def lat_lon_to_xyz_vector(latlon, radius):
        latlon = np.radians(latlon)
        x = -radius * math.sin(latlon[0]) * math.cos(latlon[1])
        y = -radius * math.cos(latlon[0])
        z = -radius * math.sin(latlon[0]) * math.sin(latlon[1])
        return np.array([x, y, z])

    @staticmethod
    def xyz_to_lat_lon(pos, radius):
        lat = math.pi - math.acos(pos[1] / radius)
        lon = math.atan2(pos[2], pos[0]) + math.pi
        return np.degrees(np.array([lat, lon]))

    @staticmethod
    def xyz_to_lat_lon_vector(pos):
        radius = np.linalg.norm(pos)
        lat = math.pi - math.acos(pos[1] / radius)
        lon = math.atan2(pos[2], pos[0]) + math.pi
        return np.degrees(np.array([lat, lon]))

    @staticmethod
    def xyz_to_uv(pos):
        radius = np.linalg.norm(pos)
        lat = math.pi - math.acos(pos[1] / radius)
        lon = math.atan2(pos[2], pos[0]) + math.pi
        return np.array([lon * MathFunctions.TwoPIInv, lat * MathFunctions.PIInv])

    @staticmethod
    def cubic_interpolation(n0, n1, n2, n3, a):
        return n1 + 0.5 * a * (n2 - n0 + a * (2 * n0 - 5 * n1 + 4 * n2 - n3 + a * (3 * (n1 - n2) + n3 - n0)))

    @staticmethod
    def rotate_around_point(point, pivot, rotation):
        dir = point - pivot
        dir = rotation.apply(dir)
        return dir + pivot


class NodeEditorWindow:
    _node_tint = None
    _node_width = None
    _node_types = None

    @property
    def node_tint(self):
        if self._node_tint is None:
            self._node_tint = self.get_node_tint()
        return self._node_tint

    @property
    def node_width(self):
        if self._node_width is None:
            self._node_width = self.get_node_width()
        return self._node_width

    @property
    def node_types(self):
        if self._node_types is None:
            self._node_types = self.get_node_types()
        return self._node_types

    def __init__(self):
        self._is_docked = None

    @property
    def is_docked(self):
        if self._is_docked is None:
            self._is_docked = lambda: False
        return self._is_docked

    @staticmethod
    def get_node_types():
        return []

    @staticmethod
    def get_node_tint():
        tints = {}
        for node_type in NodeEditorWindow.get_node_types():
            attribs = getattr(node_type, 'NodeTintAttribute', None)
            if attribs:
                tints[node_type] = attribs.color
        return tints

    @staticmethod
    def get_node_width():
        widths = {}
        for node_type in NodeEditorWindow.get_node_types():
            attribs = getattr(node_type, 'NodeWidthAttribute', None)
            if attribs:
                widths[node_type] = attribs.width
        return widths

    @staticmethod
    def get_field_info(type_, field_name):
        try:
            field = getattr(type_, field_name)
        except AttributeError:
            field = None
            while type_ and type_ != Node:
                type_ = type_.__base__
                field = getattr(type_, field_name, None)
                if field:
                    break
        return field

    @staticmethod
    def get_derived_types(base_type):
        return [cls for cls in base_type.__subclasses__() if not getattr(cls, '__abstractmethods__', False)]

    @staticmethod
    def add_custom_context_menu_items(context_menu, obj):
        items = NodeEditorWindow.get_context_menu_methods(obj)
        if items:
            context_menu.append_separator()
            for context_menu, method in items:
                context_menu.append_item(context_menu.menu_item, lambda: method(obj))

    @staticmethod
    def get_context_menu_methods(obj):
        type_ = type(obj)
        methods = [method for method in dir(type_) if callable(getattr(type_, method))]
        kvp = []
        for method_name in methods:
            method = getattr(type_, method_name)
            attribs = getattr(method, 'ContextMenu', None)
            if attribs:
                if len(method.__code__.co_varnames) > 1:
                    print(f"Method {type_.__name__}.{method.__name__} has parameters and cannot be used for context menu commands.")
                    continue
                if getattr(method, '__self__', None) is None:
                    print(f"Method {type_.__name__}.{method.__name__} is static and cannot be used for context menu commands.")
                    continue
                for attrib in attribs:
                    kvp.append((attrib, method))
        return kvp

    @staticmethod
    def open_preferences():
        try:
            print("Preferences window opened")
        except Exception as e:
            print(e)
            print("Unity has changed around internally. Can't open properties through reflection. Please contact xNode developer and supply unity version number.")



import numpy as np
import random

class NodeEditorWindow:
    _node_tint = None
    _node_width = None
    _node_types = None

    @property
    def node_tint(self):
        if self._node_tint is None:
            self._node_tint = self.get_node_tint()
        return self._node_tint

    @property
    def node_width(self):
        if self._node_width is None:
            self._node_width = self.get_node_width()
        return self._node_width

    @property
    def node_types(self):
        if self._node_types is None:
            self._node_types = self.get_node_types()
        return self._node_types

    def __init__(self):
        self._is_docked = None

    @property
    def is_docked(self):
        if self._is_docked is None:
            # Simulating access to a 'docked' property
            self._is_docked = lambda: False
        return self._is_docked

    @staticmethod
    def get_node_types():
        # Simulate getting all classes deriving from Node via reflection
        return []

    @staticmethod
    def get_node_tint():
        tints = {}
        for node_type in NodeEditorWindow.get_node_types():
            attribs = getattr(node_type, 'NodeTintAttribute', None)
            if attribs:
                tints[node_type] = attribs.color
        return tints

    @staticmethod
    def get_node_width():
        widths = {}
        for node_type in NodeEditorWindow.get_node_types():
            attribs = getattr(node_type, 'NodeWidthAttribute', None)
            if attribs:
                widths[node_type] = attribs.width
        return widths

    @staticmethod
    def get_field_info(type_, field_name):
        # Simulate getting FieldInfo of a field, including those that are private and/or inherited
        try:
            field = getattr(type_, field_name)
        except AttributeError:
            field = None
            while type_ and type_ != Node:
                type_ = type_.__base__
                field = getattr(type_, field_name, None)
                if field:
                    break
        return field

    @staticmethod
    def get_derived_types(base_type):
        # Simulate getting all classes deriving from baseType via reflection
        return [cls for cls in base_type.__subclasses__() if not getattr(cls, '__abstractmethods__', False)]

    @staticmethod
    def add_custom_context_menu_items(context_menu, obj):
        items = NodeEditorWindow.get_context_menu_methods(obj)
        if items:
            context_menu.append_separator()
            for context_menu, method in items:
                context_menu.append_item(context_menu.menu_item, lambda: method(obj))

    @staticmethod
    def get_context_menu_methods(obj):
        type_ = type(obj)
        methods = [method for method in dir(type_) if callable(getattr(type_, method))]
        kvp = []
        for method_name in methods:
            method = getattr(type_, method_name)
            attribs = getattr(method, 'ContextMenu', None)
            if attribs:
                if len(method.__code__.co_varnames) > 1:
                    print(f"Method {type_.__name__}.{method.__name__} has parameters and cannot be used for context menu commands.")
                    continue
                if getattr(method, '__self__', None) is None:
                    print(f"Method {type_.__name__}.{method.__name__} is static and cannot be used for context menu commands.")
                    continue
                for attrib in attribs:
                    kvp.append((attrib, method))
        return kvp

    @staticmethod
    def open_preferences():
        try:
            # Simulating opening the preferences window
            print("Preferences window opened")
        except Exception as e:
            print(e)
            print("Unity has changed around internally. Can't open properties through reflection. Please contact xNode developer and supply unity version number.")

class FoliageRenderer:
    def __init__(self):
        self.detail_objects = []
        self.spawned_prefabs = []
        self.matrices = []
        self.planet = None
        self.quad = None
        self.old_position = np.array([0, 0, 0])
        self.position = np.array([0, 0, 0])
        self.old_rotation = np.array([0, 0, 0, 0])
        self.rotation = np.array([0, 0, 0, 0])
        self.initialized = False
        self.generating = False

    def initialize(self):
        self.position = self.transform.position
        self.rotation = self.transform.rotation
        self.matrices = [self.to_matrix_4x4_array(detail.pos_rots, detail.mesh_scale) for detail in self.detail_objects]
        self.old_position = self.position
        self.old_rotation = self.rotation
        self.initialized = True

    def update(self):
        if self.initialized:
            self.position = self.transform.position
            self.rotation = self.transform.rotation

            if not np.array_equal(self.position, self.old_position) or not np.array_equal(self.rotation, self.old_rotation):
                self.recalculate_matrices()
                self.old_position = self.position
                self.old_rotation = self.rotation

            for i, matrix in enumerate(self.matrices):
                if not self.detail_objects[i].use_gpu_instancing:
                    for j, mat in enumerate(matrix):
                        self.graphics_draw_mesh(self.detail_objects[i].mesh, mat, self.detail_objects[i].material, 0)
                else:
                    self.graphics_draw_mesh_instanced(self.detail_objects[i].mesh, 0, self.detail_objects[i].material, matrix)
        elif not self.generating and self.planet.detail_objects_generating < self.planet.detail_objects_generating_simultaneously:
            self.generating = True
            self.start_coroutine(self.generate_details())
            self.planet.detail_objects_generating += 1

    def recalculate_matrices(self):
        for i in range(len(self.matrices)):
            self.matrices[i] = self.to_matrix_4x4_array(self.detail_objects[i].pos_rots, self.detail_objects[i].mesh_scale)

    def to_matrix_4x4_array(self, pos_rots, mesh_scale):
        matrices = []
        if not np.array_equal(self.rotation, np.array([0, 0, 0, 1])):
            for pos_rot in pos_rots:
                matrices.append(self.set_trs(self.rotate_around_point(pos_rot.position + self.position, self.position, self.rotation), pos_rot.rotation * self.rotation, mesh_scale))
        else:
            for pos_rot in pos_rots:
                matrices.append(self.set_trs(pos_rot.position + self.transform.position, pos_rot.rotation, mesh_scale))
        return matrices

    def on_destroy(self):
        if self.generating:
            self.planet.detail_objects_generating -= 1

        while self.spawned_prefabs:
            self.destroy(self.spawned_prefabs[0])
            self.spawned_prefabs.pop(0)

    def generate_details(self):
        pass  # Coroutine to generate details

    # Placeholder methods
    def transform(self):
        return np.array([0, 0, 0]), np.array([0, 0, 0, 0])

    def graphics_draw_mesh(self, mesh, matrix, material, layer):
        pass

    def graphics_draw_mesh_instanced(self, mesh, submesh_index, material, matrices):
        pass

    def set_trs(self, position, rotation, scale):
        return np.array([position, rotation, scale])

    def rotate_around_point(self, point, pivot, rotation):
        return point

    def destroy(self, obj):
        pass

    def start_coroutine(self, coroutine):
        pass

class FoliageGenerator:
    def __init__(self, quad, mesh, rotation, down, foliage_biomes, seed):
        self.positions = []
        self.normals = []
        self.indices = []
        self.position = np.array([0, 0, 0])
        self.mesh = mesh
        self.down = down
        self.rotation = rotation
        self.foliage_biomes = foliage_biomes
        self.random = random.Random(seed)
        self.mesh_tris = quad.planet.quad_arrays.tris0000
        self.mesh_vertices = mesh.vertices
        self.mesh_normals = mesh.normals
        self.mesh_colors = mesh.colors32
        self.mesh_uv4 = mesh.uv4

    def point_cloud(self, number):
        num_tris = len(self.mesh_tris) // 3
        for _ in range(number):
            random_triangle = self.random.randint(0, num_tris - 1) * 3

            if self.foliage_biomes != 'All':
                col = self.mesh_colors[self.mesh_tris[random_triangle]]
                uv = self.mesh_uv4[self.mesh_tris[random_triangle]]

                if (self.foliage_biomes & 1) and col[0] > 127:
                    pass
                elif (self.foliage_biomes & 2) and col[1] > 127:
                    pass
                elif (self.foliage_biomes & 4) and col[2] > 127:
                    pass
                elif (self.foliage_biomes & 8) and col[3] > 127:
                    pass
                elif (self.foliage_biomes & 16) and uv[0] > 0.5:
                    pass
                elif (self.foliage_biomes & 32) and uv[1] > 0.5:
                    pass
                else:
                    continue

            a = self.mesh_vertices[self.mesh_tris[random_triangle]]
            b = self.mesh_vertices[self.mesh_tris[random_triangle + 1]]
            c = self.mesh_vertices[self.mesh_tris[random_triangle + 2]]

            x = self.random.random()
            y = self.random.random()

            if x + y >= 1:
                x = 1 - x
                y = 1 - y

            point_on_mesh = a + x * (b - a) + y * (c - a)

            self.indices.append(len(self.indices))
            self.positions.append(point_on_mesh)
            self.normals.append(self.mesh_normals[self.mesh_tris[random_triangle]])

    def positions(self, number, detail_object):
        if number > len(self.positions):
            number = len(self.positions)

        i = len(self.positions) - number
        pos_rots = []

        rdown = self.rotation * self.down
        rot = np.quaternion(90, 0, 0) * np.quaternion(self.down)

        pos = rdown * -detail_object.mesh_offset_up
        j = 0
        for pos in self.positions[i:]:
            pos_rots.append(PosRot(pos + pos, rot))
            j += 1

        self.positions = self.positions[:-number]
        self.indices = self.indices[:-number]
        self.normals = self.normals[:-number]

        return pos_rots

    def create_mesh(self):
        mesh = Mesh()
        mesh.set_vertices(self.positions)
        mesh.set_indices(self.indices, 'Points')
        mesh.set_normals(self.normals)

        self.positions = []
        self.normals = []
        self.indices = []

        return mesh

class PosRot:
    def __init__(self, position, rotation):
        self.position = position
        self.rotation = rotation

class DetailObject:
    def __init__(self):
        self.number = 0
        self.mesh_offset_up = 0.0
        self.pos_rots = []

class DetailMesh(DetailObject):
    def __init__(self, mesh=None, material=None):
        super().__init__()
        self.mesh = mesh
        self.material = material
        self.mesh_scale = np.array([1.0, 1.0, 1.0])
        self.use_gpu_instancing = False
        self.is_grass = False

    def __init__(self, detail_mesh):
        super().__init__()
        self.mesh = detail_mesh.mesh
        self.material = detail_mesh.material
        self.number = detail_mesh.number
        self.mesh_offset_up = detail_mesh.mesh_offset_up
        self.mesh_scale = detail_mesh.mesh_scale
        self.use_gpu_instancing = detail_mesh.use_gpu_instancing
        self.is_grass = detail_mesh.is_grass
        self.pos_rots = []

class DetailPrefab(DetailObject):
    def __init__(self, prefab=None):
        super().__init__()
        self.prefab = prefab

    def instantiate_objects(self):
        objects = []
        for pos_rot in self.pos_rots:
            objects.append(self.instantiate(self.prefab, pos_rot.position, pos_rot.rotation))
        return objects

class Mesh:
    def __init__(self):
        self.vertices = []
        self.normals = []
        self.colors32 = []
        self.uv4 = []
        self.index_format = 'UInt32'

    def set_vertices(self, vertices):
        self.vertices = vertices

    def set_indices(self, indices, topology):
        self.indices = indices

    def set_normals(self, normals):
        self.normals = normals

class Quaternion:
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __mul__(self, other):
        return Quaternion(
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        )

# Example usage
window = NodeEditorWindow()
window.open_preferences()


import numpy as np

class NodeEditorWindow:
    current = None

    def __init__(self):
        self._port_connection_points = {}
        self._references = []
        self._rects = []
        self._node_sizes = {}
        self.graph = None
        self._pan_offset = np.array([0.0, 0.0])
        self._zoom = 1.0

    @property
    def port_connection_points(self):
        return self._port_connection_points

    @property
    def node_sizes(self):
        return self._node_sizes

    @property
    def pan_offset(self):
        return self._pan_offset

    @pan_offset.setter
    def pan_offset(self, value):
        self._pan_offset = value
        self.repaint()

    @property
    def zoom(self):
        return self._zoom

    @zoom.setter
    def zoom(self, value):
        self._zoom = max(1.0, min(value, 5.0))
        self.repaint()

    def on_disable(self):
        # Cache portConnectionPoints before serialization starts
        count = len(self.port_connection_points)
        self._references = [NodePortReference(key) for key in self.port_connection_points.keys()]
        self._rects = list(self.port_connection_points.values())

    def on_enable(self):
        # Reload portConnectionPoints if there are any
        if len(self._references) == len(self._rects):
            for ref, rect in zip(self._references, self._rects):
                node_port = ref.get_node_port()
                if node_port:
                    self._port_connection_points[node_port] = rect

    def on_focus(self):
        NodeEditorWindow.current = self
        self.graph_editor = self.get_editor(self.graph)
        if self.graph_editor and NodeEditorPreferences.get_settings().auto_save:
            self.save_assets()

    @staticmethod
    def init():
        window = NodeEditorWindow()
        window.title = "xNode"
        window.wants_mouse_move = True
        window.show()
        return window

    def save(self):
        if self.contains_asset(self.graph):
            self.set_dirty(self.graph)
            if NodeEditorPreferences.get_settings().auto_save:
                self.save_assets()
        else:
            self.save_as()

    def save_as(self):
        path = self.save_file_panel("Save NodeGraph", "NewNodeGraph", "asset")
        if not path:
            return
        existing_graph = self.load_asset_at_path(path, NodeGraph)
        if existing_graph:
            self.delete_asset(path)
        self.create_asset(self.graph, path)
        self.set_dirty(self.graph)
        if NodeEditorPreferences.get_settings().auto_save:
            self.save_assets()

    def draggable_window(self, window_id):
        pass  # Placeholder for GUI.DragWindow()

    def window_to_grid_position(self, window_position):
        return (window_position - self.position_size * 0.5 - self.pan_offset / self.zoom) * self.zoom

    def grid_to_window_position(self, grid_position):
        return self.position_size * 0.5 + self.pan_offset / self.zoom + grid_position / self.zoom

    def grid_to_window_rect_no_clipped(self, grid_rect):
        grid_rect.position = self.grid_to_window_position_no_clipped(grid_rect.position)
        return grid_rect

    def grid_to_window_rect(self, grid_rect):
        grid_rect.position = self.grid_to_window_position(grid_rect.position)
        grid_rect.size /= self.zoom
        return grid_rect

    def grid_to_window_position_no_clipped(self, grid_position):
        center = self.position_size * 0.5
        x_offset = center[0] * self.zoom + (self.pan_offset[0] + grid_position[0])
        y_offset = center[1] * self.zoom + (self.pan_offset[1] + grid_position[1])
        return np.array([x_offset, y_offset])

    def select_node(self, node, add):
        if add:
            selection = list(self.selection_objects)
            selection.append(node)
            self.selection_objects = selection
        else:
            self.selection_objects = [node]

    def deselect_node(self, node):
        selection = list(self.selection_objects)
        selection.remove(node)
        self.selection_objects = selection

    @staticmethod
    def on_open(instance_id, line):
        node_graph = NodeEditorWindow.instance_id_to_object(instance_id, NodeGraph)
        if node_graph:
            window = NodeEditorWindow.get_window("xNode")
            window.wants_mouse_move = True
            window.graph = node_graph
            return True
        return False

    @staticmethod
    def repaint_all():
        windows = NodeEditorWindow.find_objects_of_type(NodeEditorWindow)
        for window in windows:
            window.repaint()

    # Placeholder methods for Unity Editor functionalities
    def repaint(self):
        pass

    def show(self):
        pass

    @property
    def position_size(self):
        return np.array([800, 600])  # Placeholder for window size

    @staticmethod
    def contains_asset(asset):
        return True

    @staticmethod
    def set_dirty(asset):
        pass

    @staticmethod
    def save_assets():
        pass

    @staticmethod
    def save_file_panel(title, default_name, extension):
        return ""

    @staticmethod
    def load_asset_at_path(path, asset_type=None):
        return None

    @staticmethod
    def delete_asset(path):
        pass

    @staticmethod
    def create_asset(asset, path):
        pass

    @property
    def selection_objects(self):
        return []

    @selection_objects.setter
    def selection_objects(self, value):
        pass

    @staticmethod
    def instance_id_to_object(instance_id, object_type=None):
        return None

    @staticmethod
    def get_window(title):
        return NodeEditorWindow()

    @staticmethod
    def find_objects_of_type(object_type):
        return []

    def get_editor(self, graph):
        return None

class NodePortReference:
    def __init__(self, node_port):
        self._node = node_port.node
        self._name = node_port.field_name

    def get_node_port(self):
        if not self._node:
            return None
        return self._node.get_port(self._name)

class NodeEditorPreferences:
    @staticmethod
    def get_settings():
        return NodeEditorPreferences()

    @property
    def auto_save(self):
        return True

class RenderTexture:
    def __init__(self, width, height, depth, dimension):
        self.width = width
        self.height = height
        self.depth = depth
        self.dimension = dimension
        self.enable_random_write = False
        self.created = False

    def is_created(self):
        return self.created

    def create(self):
        self.created = True

class ComputeBuffer:
    def __init__(self, count, stride):
        self.count = count
        self.stride = stride

    def set_data(self, data):
        pass

class ComputeShader:
    def __init__(self):
        self.kernels = {}

    def find_kernel(self, name):
        return self.kernels.get(name, -1)

    def set_texture(self, kernel, name, texture):
        pass

    def set_buffer(self, kernel, name, buffer):
        pass

    def set_int(self, name, value):
        pass

    def dispatch(self, kernel, x, y, z):
        pass

class CBWrite:
    write_names_2d = [
        ["write2DC1", "_Des2DC1", "_Buffer2DC1"],
        ["write2DC2", "_Des2DC2", "_Buffer2DC2"],
        ["write2DC3", "_Des2DC3", "_Buffer2DC3"],
        ["write2DC4", "_Des2DC4", "_Buffer2DC4"]
    ]

    write_names_3d = [
        ["write3DC1", "_Des3DC1", "_Buffer3DC1"],
        ["write3DC2", "_Des3DC2", "_Buffer3DC2"],
        ["write3DC3", "_Des3DC3", "_Buffer3DC3"],
        ["write3DC4", "_Des3DC4", "_Buffer3DC4"]
    ]

    @staticmethod
    def into_render_texture(tex, channels, buffer, write):
        CBWrite.check(tex, channels, buffer, write)

        kernel = -1
        depth = 1

        if tex.dimension == "Tex3D":
            depth = tex.depth
            kernel = write.find_kernel(CBWrite.write_names_3d[channels - 1][0])
            write.set_texture(kernel, CBWrite.write_names_3d[channels - 1][1], tex)
            write.set_buffer(kernel, CBWrite.write_names_3d[channels - 1][2], buffer)
        else:
            kernel = write.find_kernel(CBWrite.write_names_2d[channels - 1][0])
            write.set_texture(kernel, CBWrite.write_names_2d[channels - 1][1], tex)
            write.set_buffer(kernel, CBWrite.write_names_2d[channels - 1][2], buffer)

        if kernel == -1:
            raise ValueError("Could not find kernel " + CBWrite.write_names_2d[channels - 1][0])

        width = tex.width
        height = tex.height

        write.set_int("_Width", width)
        write.set_int("_Height", height)
        write.set_int("_Depth", depth)

        pad_x = 0 if width % 8 == 0 else 1
        pad_y = 0 if height % 8 == 0 else 1
        pad_z = 0 if depth % 8 == 0 else 1

        write.dispatch(kernel, max(1, width // 8 + pad_x), max(1, height // 8 + pad_y), max(1, depth // 8 + pad_z))

    @staticmethod
    def into_render_texture_with_path(tex, channels, path, buffer, write):
        CBWrite.check(tex, channels, buffer, write)

        kernel = -1
        depth = 1

        if tex.dimension == "Tex3D":
            depth = tex.depth
            kernel = write.find_kernel(CBWrite.write_names_3d[channels - 1][0])
            write.set_texture(kernel, CBWrite.write_names_3d[channels - 1][1], tex)
            write.set_buffer(kernel, CBWrite.write_names_3d[channels - 1][2], buffer)
        else:
            kernel = write.find_kernel(CBWrite.write_names_2d[channels - 1][0])
            write.set_texture(kernel, CBWrite.write_names_2d[channels - 1][1], tex)
            write.set_buffer(kernel, CBWrite.write_names_2d[channels - 1][2], buffer)

        if kernel == -1:
            raise ValueError("Could not find kernel " + CBWrite.write_names_2d[channels - 1][0])

        width = tex.width
        height = tex.height
        size = width * height * depth * channels

        map_data = CBWrite.load_raw_file(path, size)
        buffer.set_data(map_data)

        write.set_int("_Width", width)
        write.set_int("_Height", height)
        write.set_int("_Depth", depth)

        pad_x = 0 if width % 8 == 0 else 1
        pad_y = 0 if height % 8 == 0 else 1
        pad_z = 0 if depth % 8 == 0 else 1

        write.dispatch(kernel, max(1, width // 8 + pad_x), max(1, height // 8 + pad_y), max(1, depth // 8 + pad_z))

    @staticmethod
    def load_raw_file(path, size):
        with open(path, "rb") as f:
            data = f.read()

        if size > len(data) // 4:
            raise ValueError(f"Raw file is not the required size ({path})")

        map_data = [struct.unpack('f', data[i:i+4])[0] for i in range(0, len(data), 4)]
        return map_data

    @staticmethod
    def check(tex, channels, buffer, write):
        if tex is None:
            raise ValueError("RenderTexture is null")

        if buffer is None:
            raise ValueError("Buffer is null")

        if write is None:
            raise ValueError("Compute shader is null")

        if channels < 1 or channels > 4:
            raise ValueError("Channels must be 1, 2, 3, or 4")

        if not tex.enable_random_write:
            raise ValueError("You must enable random write on render texture")

        if not tex.is_created():
            raise ValueError("Tex has not been created (Call Create() on tex)")


import numpy as np
import struct

class Heightmap:
    def __init__(self, width, height, is16bit, create_empty=True):
        self.width = width
        self.height = height
        self.is16bit = is16bit
        if is16bit:
            self.ushorts = np.zeros((height, width), dtype=np.uint16) if create_empty else None
        else:
            self.bytes = np.zeros((height, width), dtype=np.uint8) if create_empty else None

    def get_file_bytes(self):
        if self.is16bit:
            return self.ushorts.tobytes()
        else:
            return self.bytes.tobytes()

    def get_texture2d(self):
        # Placeholder for getting a texture from the heightmap
        pass

class HeightmapImporter:
    def __init__(self):
        self.path = ""
        self.filename = "heightmapPhotoshop"
        self.width = 8192
        self.height = 4096
        self.is16bit = False
        self.reverse_byte_order = False
        self.cut_tiff_header = False
        self.preview = None

    def on_gui(self):
        print("Heightmap Importer")
        self.path = input("Path: ")
        self.filename = input("Filename: ")
        self.width = int(input("Width: "))
        self.height = int(input("Height: "))
        self.is16bit = bool(input("16-bit (True/False): "))
        if self.is16bit:
            self.reverse_byte_order = bool(input("Reverse byte order (True/False): "))
        self.cut_tiff_header = bool(input("Cut header (True/False): "))
        
        if input("Convert (Yes/No): ").lower() == "yes":
            self.convert_to_heightmap()

        print("Converts grayscale Photoshop or GIMP .raw files to heightmaps. Photoshop: Export with Macintosh byte order and 0 header. GIMP: Export with Standard (R, G, B). You can also import uncompressed tiffs if you enable Cut Header.")
        print(self.preview)

    def convert_to_heightmap(self):
        with open(self.path, "rb") as fs:
            if self.cut_tiff_header:
                fs.seek(8)
                
            heightmap = None

            if self.is16bit:
                half_length = self.width * self.height
                hh = self.height - 1
                heightmap = Heightmap(self.width, self.height, True)

                if fs.seek(0, 2) != heightmap.ushorts.size * 2 and not self.cut_tiff_header:
                    print("Failed to convert to heightmap. Incorrect resolution or incompatible file format.")
                    return

                fs.seek(0)
                try:
                    for i in range(half_length):
                        index = (hh - (i // self.width)) * self.width + (i % self.width)
                        us = fs.read(2)
                        if self.reverse_byte_order:
                            heightmap.ushorts[index] = (us[1] << 8) + us[0]
                        else:
                            heightmap.ushorts[index] = (us[0] << 8) + us[1]
                except IndexError as e:
                    print("Failed to convert to heightmap. Incorrect resolution or incompatible file format.")
                    print(e)
                    return
            else:
                length = self.width * self.height
                hh = self.height - 1
                heightmap = Heightmap(self.width, self.height, False)

                if fs.seek(0, 2) != heightmap.bytes.size and not self.cut_tiff_header:
                    print("Failed to convert to heightmap. Incorrect resolution or incompatible file format.")
                    return

                fs.seek(0)
                try:
                    for i in range(length):
                        index = (hh - (i // self.width)) * self.width + (i % self.width)
                        heightmap.bytes[index] = fs.read(1)[0]
                except IndexError as e:
                    print("Failed to convert to heightmap. Incorrect resolution or incompatible file format.")
                    print(e)
                    return

            print("Successfully converted!")
            with open(self.filename + ".bytes", "wb") as f:
                f.write(heightmap.get_file_bytes())

            # Placeholder for refreshing assets
            self.preview = heightmap.get_texture2d()

# Example usage
heightmap_importer = HeightmapImporter()
heightmap_importer.on_gui()

class ModuleType:
    Select = "Select"
    Curve = "Curve"
    Blend = "Blend"
    Remap = "Remap"
    Add = "Add"
    Subtract = "Subtract"
    Multiply = "Multiply"
    Min = "Min"
    Max = "Max"
    Scale = "Scale"
    ScaleBias = "ScaleBias"
    Abs = "Abs"
    Invert = "Invert"
    Clamp = "Clamp"
    Const = "Const"
    Terrace = "Terrace"

class OperatorNodeEditor:
    no_port = ["output"]
    one_port = ["input0", "output"]
    two_ports = ["input0", "input1", "output"]
    three_ports = ["input0", "input1", "input2", "output"]

    def __init__(self):
        self.number_of_inputs = 0

    def on_body_gui(self, node):
        print("Operator Node Editor")
        temp = node.module_type
        temp = input(f"ModuleType ({temp}): ")
        node.module_type = temp

        if node.parameters is None:
            node.parameters = [0] * 6
        if node.module_type == ModuleType.Curve and node.curve is None:
            node.curve = []

        if node.module_type == ModuleType.Select:
            self.number_of_inputs = 3
            node.parameters[0] = float(input("Fall Off: "))
            node.parameters[1] = float(input("Min: "))
            node.parameters[2] = float(input("Max: "))
        elif node.module_type == ModuleType.Curve:
            self.number_of_inputs = 1
            # Placeholder for curve input
            node.curve = input("Curve: ")
        elif node.module_type == ModuleType.Blend:
            self.number_of_inputs = 2
            node.parameters[0] = float(input("Bias: "))
        elif node.module_type == ModuleType.Remap:
            self.number_of_inputs = 1
            scale = input("Scale (x, y, z): ").split(',')
            offset = input("Offset (x, y, z): ").split(',')
            node.parameters = [float(s) for s in scale] + [float(o) for o in offset]
        elif node.module_type in [ModuleType.Add, ModuleType.Subtract, ModuleType.Multiply, ModuleType.Min, ModuleType.Max]:
            self.number_of_inputs = 2
        elif node.module_type == ModuleType.Scale:
            self.number_of_inputs = 1
            node.parameters[0] = float(input("Scale: "))
        elif node.module_type == ModuleType.ScaleBias:
            self.number_of_inputs = 1
            node.parameters[0] = float(input("Scale: "))
            node.parameters[1] = float(input("Bias: "))
        elif node.module_type == ModuleType.Abs:
            self.number_of_inputs = 1
        elif node.module_type == ModuleType.Clamp:
            self.number_of_inputs = 1
            node.parameters[0] = float(input("Min: "))
            node.parameters[1] = float(input("Max: "))
        elif node.module_type == ModuleType.Invert:
            self.number_of_inputs = 1
        elif node.module_type == ModuleType.Const:
            self.number_of_inputs = 0
            node.parameters[0] = float(input("Constant: "))
        elif node.module_type == ModuleType.Terrace:
            self.number_of_inputs = 1
            # Placeholder for property field
            node.parameters = input("Parameters: ").split(',')

        if node.preview is None:
            # Placeholder for threading
            node.update_preview()

        if node.preview_changed:
            if node.preview_heightmap is None:
                return
            node.preview = node.preview_heightmap.get_texture2d()
            node.preview_changed = False
            # Placeholder for repainting

        print(node.preview)

# Example usage
class OperatorNode:
    def __init__(self):
        self.module_type = ModuleType.Const
        self.parameters = None
        self.curve = None
        self.preview = None
        self.preview_changed = False
        self.preview_heightmap = None

    def update_preview(self):
        # Placeholder for preview update
        pass

operator_node = OperatorNode()
operator_node_editor = OperatorNodeEditor()
operator_node_editor.on_body_gui(operator_node)

class QuadPool:
    def __init__(self, planet):
        self.planet = planet
        self.quads_pool = []
        self.quad_pool_max_size = 30

    def get_quad(self, tr_position, rotation):
        if len(self.quads_pool) == 0:
            return Quad(tr_position, rotation)
        else:
            q = self.quads_pool.pop(0)
            q.tr_position = tr_position
            q.rotation = rotation
            return q

    def remove_quad(self, quad):
        if quad.mesh_generator and quad.mesh_generator.is_running:
            quad.mesh_generator.dispose()
            quad.mesh_generator = None
        if quad.coroutine:
            if not quad.is_splitting:
                self.planet.detail_objects_generating -= 1
            # Simulate coroutine stopping
        if quad.mesh_generator:
            quad.mesh_generator.dispose()
            quad.mesh_generator = None
        if quad.children:
            for child in quad.children:
                self.remove_quad(child)
        if quad.rendered_quad:
            self.planet.quad_game_object_pool.remove_game_object(quad)
        quad.reset()
        self.planet.quads.remove(quad)
        self.planet.quad_split_queue.remove(quad)
        self.planet.quad_indicies.remove(quad.index)
        if len(self.quads_pool) < self.quad_pool_max_size:
            self.quads_pool.append(quad)

class QuadGameObjectPool:
    def __init__(self, planet):
        self.planet = planet
        self.quad_go_pool = []
        self.rendered_quad_pool_max_size = 30

    def get_game_object(self, quad):
        rquad = None
        rotation = self.planet.rotation
        pl_pos = self.planet.transform.position

        if len(self.quad_go_pool) == 0:
            rquad = GameObject("Quad")
            rquad.transform.position = rotation * quad.tr_position + rotation * quad.mesh_offset + pl_pos
            rquad.transform.rotation = rotation
            rquad.add_component(MeshRenderer).material = self.planet.planet_material
            if self.planet.hide_quads:
                rquad.hide_flags = True
        else:
            rquad = self.quad_go_pool.pop(-1)
            if rquad is None:
                rquad = GameObject("Quad")
                rquad.transform.position = rotation * quad.tr_position + rotation * quad.mesh_offset + pl_pos
                rquad.transform.rotation = rotation
                rquad.add_component(MeshRenderer).material = self.planet.planet_material
                if self.planet.hide_quads:
                    rquad.hide_flags = True
            rquad.transform.position = rotation * quad.tr_position + rotation * quad.mesh_offset + pl_pos
            rquad.transform.rotation = rotation

        rquad.get_component(MeshFilter).mesh = quad.mesh
        rquad.name = f"Quad {quad.index}"

        if self.planet.generate_colliders[quad.level]:
            rquad.add_component(MeshCollider).convex = True

        return rquad

    def remove_game_object(self, quad):
        if len(self.quad_go_pool) < self.rendered_quad_pool_max_size:
            quad.rendered_quad.remove_component(PlanetaryTerrain.Foliage.FoliageRenderer)
            quad.rendered_quad.remove_component(MeshCollider)
            self.quad_go_pool.append(quad.rendered_quad)
            quad.rendered_quad.set_active(False)
            quad.rendered_quad = None
        else:
            del quad.rendered_quad

# Placeholder classes for missing components
class MeshRenderer:
    pass

class MeshFilter:
    pass

class MeshCollider:
    pass

# Placeholder for MonoBehaviour
class MonoBehaviour:
    @staticmethod
    def destroy(obj):
        del obj

# Placeholder for Camera and Application classes
class Vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __mul__(self, other):
        return Vector3(self.x * other, self.y * other, self.z * other)

    def __repr__(self):
        return f"Vector3(x={self.x}, y={self.y}, z={self.z})"

class Transform:
    def __init__(self, position):
        self.position = position

class Camera:
    # Provide a default position for the main camera
    main = Transform(Vector3(0, 0, 0))

class Application:
    dataPath = "/path/to/application/data"

# Example usage
print(Camera.main.position)  # Output: Vector3(x=0, y=0, z=0)
print(Application.dataPath)  # Output: /path/to/application/data

# Placeholder for AssetDatabase
class AssetDatabase:
    @staticmethod
    def refresh():
        pass

# Example usage
planet = Planet(radius=6371, scaled_space_factor=1)
sun = GameObject("Sun")
atmosphere = Atmosphere(planet, sun)
quad_pool = QuadPool(planet)
quad_game_object_pool = QuadGameObjectPool(planet)

import numpy as np
import math

class Vector3:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Vector3({self.x}, {self.y}, {self.z})"

    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

class Quaternion:
    pass

class QuaternionD(Quaternion):
    pass

class Vector3d:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Vector3d({self.x}, {self.y}, {self.z})"

    def to_vector3(self):
        return Vector3(self.x, self.y, self.z)

class GameObject:
    def __init__(self, name="GameObject"):
        self.name = name
        self.transform = Transform()
        self.components = {}
        self.active = True

    def add_component(self, component_type):
        component = component_type()
        self.components[component_type] = component
        return component

    def get_component(self, component_type):
        return self.components.get(component_type, None)

    def set_active(self, active):
        self.active = active

class Transform:
    def __init__(self):
        self.position = Vector3()
        self.rotation = Quaternion()
        self.local_position = Vector3()
        self.local_scale = Vector3(1, 1, 1)
        self.parent = None

    def set_parent(self, parent):
        self.parent = parent

class Material:
    def __init__(self):
        self.properties = {}

    def set_vector(self, name, value):
        self.properties[name] = value

    def set_float(self, name, value):
        self.properties[name] = value

class Planet:
    def __init__(self, radius, scaled_space_factor):
        self.radius = radius
        self.scaled_space_factor = scaled_space_factor
        self.scaled_space_copy = None
        self.quad_game_object_pool = QuadGameObjectPool(self)
        self.quads = []
        self.quad_split_queue = []
        self.quad_indicies = []
        self.transform = Transform()

class Quad:
    def __init__(self, tr_position, rotation):
        self.tr_position = tr_position
        self.rotation = rotation
        self.mesh_generator = None
        self.coroutine = None
        self.children = None
        self.rendered_quad = None
        self.mesh = None
        self.index = None
        self.is_splitting = False

    def reset(self):
        pass

class Atmosphere:
    def __init__(self, planet, sun):
        self.m_sun = sun
        self.m_sky_from_space = Material()
        self.m_sky_from_atmosphere = Material()
        self.m_hdr_exposure = 0.8
        self.m_wave_length = Vector3(0.65, 0.57, 0.475)
        self.m_e_sun = 8.0
        self.m_kr = 0.0025
        self.m_km = 0.0010
        self.m_g = -0.990
        self.atmosphere_layer = 8
        self.m_outer_scale_factor = 1.025
        self.m_inner_radius = 0
        self.m_outer_radius = 0
        self.m_scale_depth = 0.25
        self.atmosphere_space = None
        self.atmosphere_ground = None
        self.planet = planet
        self.position = Vector3()
        self.scaled_space_copy = False

        self.start()

    def start(self):
        self.scaled_space_copy = self.planet.scaled_space_copy is not None

        if self.scaled_space_copy:
            self.position = self.planet.scaled_space_copy.transform.position
            radius = self.planet.radius / self.planet.scaled_space_factor
        else:
            radius = self.planet.radius

        self.m_inner_radius = radius
        self.m_outer_radius = self.m_outer_scale_factor * radius

        self.atmosphere_space = GameObject("AtmosphereSpace")
        self.atmosphere_space.transform.position = self.position if self.scaled_space_copy else self.planet.transform.position
        self.atmosphere_space.transform.local_scale = Vector3(self.m_outer_radius, self.m_outer_radius, self.m_outer_radius)

        self.atmosphere_ground = GameObject("AtmosphereGround")
        self.atmosphere_ground.transform.position = self.position if self.scaled_space_copy else self.planet.transform.position
        self.atmosphere_ground.transform.local_scale = Vector3(self.m_outer_radius, self.m_outer_radius, self.m_outer_radius)
        self.atmosphere_ground.set_active(False)

        self.m_sky_from_space = Material()
        self.m_sky_from_atmosphere = Material()

        if self.scaled_space_copy:
            self.atmosphere_ground.layer = self.atmosphere_layer
            self.atmosphere_space.layer = self.atmosphere_layer
        self.init_material(self.m_sky_from_space)
        self.init_material(self.m_sky_from_atmosphere)

    def update(self):
        if self.distance(self.planet.transform.position, Camera.main.transform.position) < (self.m_outer_radius * self.planet.scaled_space_factor if self.scaled_space_copy else self.m_outer_radius):
            self.atmosphere_ground.set_active(True)
            self.atmosphere_space.set_active(False)
        else:
            self.atmosphere_ground.set_active(False)
            self.atmosphere_space.set_active(True)

        if self.scaled_space_copy:
            self.atmosphere_ground.transform.parent = self.planet.scaled_space_copy.transform
            self.atmosphere_space.transform.parent = self.planet.scaled_space_copy.transform
            self.atmosphere_ground.transform.local_position = Vector3()
            self.atmosphere_space.transform.local_position = Vector3()

        self.init_material(self.m_sky_from_space)
        self.init_material(self.m_sky_from_atmosphere)

    class Vector3:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

        def __iter__(self):
            return iter((self.x, self.y, self.z))

        def __mul__(self, other):
            return Vector3(self.x * other, self.y * other, self.z * other)

        def __repr__(self):
            return f"Vector3(x={self.x}, y={self.y}, z={self.z})"

    class Material:
        def set_vector(self, name, vector):
            print(f"Setting vector {name} to {vector}")

        def set_float(self, name, value):
            print(f"Setting float {name} to {value}")

    class Transform:
        def __init__(self, position):
            self.position = position

    class Sun:
        def __init__(self, position):
            self.transform = Transform(position)

    class Planet:
        def __init__(self, position, scaled_space_copy=None):
            self.transform = Transform(position)
            self.scaled_space_copy = scaled_space_copy

    class Atmosphere:
        def __init__(self):
            self.m_wave_length = Vector3(650, 570, 475)
            self.m_outer_radius = 10.0
            self.m_inner_radius = 5.0
            self.m_kr = 0.0025
            self.m_km = 0.0015
            self.m_e_sun = 20.0
            self.m_scale_depth = 0.25
            self.m_hdr_exposure = 1.3
            self.m_g = -0.95
            self.m_sun = Sun(Vector3(0, 10, 20))
            self.planet = Planet(Vector3(0, 0, 0), Transform(Vector3(1, 2, 3)))

        def init_material(self, mat):
            inv_wave_length4 = Vector3(1.0 / self.m_wave_length.x**4, 1.0 / self.m_wave_length.y**4, 1.0 / self.m_wave_length.z**4)
            scale = 1.0 / (self.m_outer_radius - self.m_inner_radius)

            mat.set_vector("v3LightPos", Vector3(*self.m_sun.transform.position) * -1.0)
            mat.set_vector("v3InvWavelength", inv_wave_length4)
            mat.set_float("fOuterRadius", self.m_outer_radius)
            mat.set_float("fOuterRadius2", self.m_outer_radius**2)
            mat.set_float("fInnerRadius", self.m_inner_radius)
            mat.set_float("fInnerRadius2", self.m_inner_radius**2)
            mat.set_float("fKrESun", self.m_kr * self.m_e_sun)
            mat.set_float("fKmESun", self.m_km * self.m_e_sun)
            mat.set_float("fKr4PI", self.m_kr * 4.0 * math.pi)
            mat.set_float("fKm4PI", self.m_km * 4.0 * math.pi)
            mat.set_float("fScale", scale)
            mat.set_float("fScaleDepth", self.m_scale_depth)
            mat.set_float("fScaleOverScaleDepth", scale / self.m_scale_depth)
            mat.set_float("fHdrExposure", self.m_hdr_exposure)
            mat.set_float("g", self.m_g)
            mat.set_float("g2", self.m_g**2)
            mat.set_vector("v3LightPos", Vector3(*self.m_sun.transform.position) * -1.0)
            mat.set_vector("v3Translate", Vector3(*self.planet.scaled_space_copy.transform.position) if self.planet.scaled_space_copy else Vector3(*self.planet.transform.position))

    import math

    # Initialize the Atmosphere and Material classes
    atmosphere = Atmosphere()
    material = Material()

    # Call the init_material method
    atmosphere.init_material(material)

    def distance(self, pos1, pos2):
        return math.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2 + (pos1.z - pos2.z)**2)


import os
from typing import List

class NodeEditorAssetModProcessor:
    @staticmethod
    def on_will_delete_asset(path, options):
        # Get the object that is requested for deletion
        obj = NodeEditorAssetModProcessor.load_asset_at_path(path)

        # If we aren't deleting a script, return
        if not isinstance(obj, MonoScript):
            return "DidNotDelete"

        # Check script type. Return if deleting a non-node script
        script = obj
        script_type = script.get_class()
        if script_type is None or (script_type != Node and not issubclass(script_type, Node)):
            return "DidNotDelete"

        # Find all ScriptableObjects using this script
        guids = NodeEditorAssetModProcessor.find_assets("t:" + script_type.__name__)
        for guid in guids:
            assetpath = NodeEditorAssetModProcessor.guid_to_asset_path(guid)
            objs = NodeEditorAssetModProcessor.load_all_asset_representations_at_path(assetpath)
            for node in objs:
                if isinstance(node, Node) and node.__class__ == script_type:
                    if node and node.graph:
                        # Delete the node and notify the user
                        print(f"{node.name} of {node.graph} depended on deleted script and has been removed automatically.")
                        node.graph.remove_node(node)
        
        # We didn't actually delete the script. Tell the internal system to carry on with normal deletion procedure
        return "DidNotDelete"

    @staticmethod
    def on_reload_editor():
        # Find all NodeGraph assets
        guids = NodeEditorAssetModProcessor.find_assets("t:NodeGraph")
        for guid in guids:
            assetpath = NodeEditorAssetModProcessor.guid_to_asset_path(guid)
            graph = NodeEditorAssetModProcessor.load_asset_at_path(assetpath, NodeGraph)
            if graph:
                graph.nodes = [node for node in graph.nodes if node]
                objs = NodeEditorAssetModProcessor.load_all_asset_representations_at_path(assetpath)
                for obj in objs:
                    if isinstance(obj, Node) and obj not in graph.nodes:
                        graph.nodes.append(obj)

    # Placeholder methods for Unity Editor functionalities
    @staticmethod
    def load_asset_at_path(path, asset_type=None):
        # Simulate loading an asset at a path
        return None

    @staticmethod
    def find_assets(filter_str):
        # Simulate finding assets
        return []

    @staticmethod
    def guid_to_asset_path(guid):
        # Simulate converting GUID to asset path
        return ""

    @staticmethod
    def load_all_asset_representations_at_path(path):
        # Simulate loading all asset representations at a path
        return []

# Simulate MonoScript and Node classes
class MonoScript:
    def get_class(self):
        # Simulate getting the class of the script
        return None

class Node:
    def __init__(self, name, graph):
        self.name = name
        self.graph = graph

class NodeGraph:
    def __init__(self):
        self.nodes = []

    def remove_node(self, node):
        self.nodes.remove(node)

# Initialize the editor reload method
NodeEditorAssetModProcessor.on_reload_editor()

class Quad:
    def __init__(self, level, distance):
        self.level = level
        self.distance = distance
        self.is_splitting = False
        self.coroutine = None
        self.has_split = False
        self.in_split_queue = False

class QuadSplitQueue:
    def __init__(self, planet):
        self.queue = []
        self.sorting_class = self.SortingClass()
        self.planet = planet
        self.stop = False

    @property
    def is_any_currently_splitting(self):
        for i in range(min(self.planet.quads_splitting_simultaneously, len(self.queue))):
            if self.queue[i].is_splitting:
                return True
        return False

    def update(self):
        if len(self.queue) > 0 and not self.stop:
            if len(self.queue) > self.planet.quads_splitting_simultaneously:
                self.queue.sort(key=self.sorting_class.compare, reverse=True)

            for i in range(self.planet.quads_splitting_simultaneously):
                if len(self.queue) > i:
                    if self.queue[i] is None:
                        self.queue.pop(i)

                    if not self.queue[i].is_splitting and self.queue[i].coroutine is None and not self.queue[i].has_split:
                        self.queue[i].coroutine = self.planet.start_coroutine(self.queue[i].split())

                    if self.queue[i].has_split:
                        self.queue[i].in_split_queue = False
                        self.queue.pop(i)
                else:
                    break
            return True
        return False

    def add(self, quad):
        if quad not in self.queue and not quad.has_split:
            self.queue.append(quad)
            quad.in_split_queue = True

    def remove(self, quad):
        if quad in self.queue and not quad.is_splitting:
            self.queue.remove(quad)
            quad.in_split_queue = False

    class SortingClass:
        @staticmethod
        def compare(x, y):
            if x.level > y.level:
                return 1
            if x.distance > y.distance and x.level == y.level:
                return 1
            return -1

# Placeholder Planet class
class Planet:
    def __init__(self):
        self.quads_splitting_simultaneously = 3

    def start_coroutine(self, coroutine):
        # Simulate starting a coroutine
        return None

# Example usage
planet = Planet()
quad_split_queue = QuadSplitQueue(planet)
quad1 = Quad(level=1, distance=100)
quad2 = Quad(level=2, distance=200)
quad_split_queue.add(quad1)
quad_split_queue.add(quad2)
quad_split_queue.update()

import math
from typing import Union
import numpy as np

class Vector3d:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __mul__(self, other: Union[float, 'Vector3d']):
        if isinstance(other, Vector3d):
            return Vector3d(self.x * other.x, self.y * other.y, self.z * other.z)
        else:
            return Vector3d(self.x * other, self.y * other, self.z * other)

    def __add__(self, other: 'Vector3d'):
        return Vector3d(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Vector3d'):
        return Vector3d(self.x - other.x, self.y - other.y, self.z - other.z)

    def __truediv__(self, other: float):
        return Vector3d(self.x / other, self.y / other, self.z / other)

    def __neg__(self):
        return Vector3d(-self.x, -self.y, -self.z)

    @property
    def sqrMagnitude(self):
        return self.x * self.x + self.y * self.y + self.z * self.z

    def Normalize(self):
        magnitude = math.sqrt(self.sqrMagnitude)
        if magnitude > 1e-6:
            self.x /= magnitude
            self.y /= magnitude
            self.z /= magnitude

    @staticmethod
    def Normalize(v):
        magnitude = math.sqrt(v.sqrMagnitude)
        if magnitude > 1e-6:
            return v / magnitude
        return v

    @staticmethod
    def Cross(a, b):
        return Vector3d(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)

    @staticmethod
    def Dot(a, b):
        return a.x * b.x + a.y * b.y + a.z * b.z

    @staticmethod
    def up():
        return Vector3d(0, 1, 0)

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

class QuaternionD:
    radToDeg = 180.0 / math.pi
    degToRad = math.pi / 180.0

    kEpsilon = np.finfo(float).eps

    def __init__(self, x=0, y=0, z=0, w=1):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    @property
    def xyz(self):
        return Vector3d(self.x, self.y, self.z)

    @xyz.setter
    def xyz(self, value):
        self.x = value.x
        self.y = value.y
        self.z = value.z

    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z
        elif index == 3:
            return self.w
        else:
            raise IndexError("Invalid Quaternion index: " + str(index) + ", can use only 0,1,2,3")

    def __setitem__(self, index, value):
        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value
        elif index == 2:
            self.z = value
        elif index == 3:
            self.w = value
        else:
            raise IndexError("Invalid Quaternion index: " + str(index) + ", can use only 0,1,2,3")

    @staticmethod
    def identity():
        return QuaternionD(0, 0, 0, 1)

    @property
    def eulerAngles(self):
        return QuaternionD.ToEulerRad(self) * self.radToDeg

    @eulerAngles.setter
    def eulerAngles(self, value):
        self = QuaternionD.FromEulerRad(value * self.degToRad)

    @property
    def Length(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w)

    @property
    def LengthSquared(self):
        return self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w

    def Set(self, new_x, new_y, new_z, new_w):
        self.x = new_x
        self.y = new_y
        self.z = new_z
        self.w = new_w

    def Normalize(self):
        scale = 1.0 / self.Length
        self.xyz = self.xyz * scale
        self.w *= scale

    @staticmethod
    def Normalize(q):
        scale = 1.0 / q.Length
        return QuaternionD(q.xyz * scale, q.w * scale)

    @staticmethod
    def Dot(a, b):
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w

    @staticmethod
    def AngleAxis(angle, axis):
        if axis.sqrMagnitude == 0.0:
            return QuaternionD.identity()
        axis.Normalize()
        radians = angle * QuaternionD.degToRad * 0.5
        axis = axis * math.sin(radians)
        return QuaternionD.Normalize(QuaternionD(axis.x, axis.y, axis.z, math.cos(radians)))

    def ToAngleAxis(self):
        angle = 2.0 * math.acos(self.w)
        den = math.sqrt(1.0 - self.w * self.w)
        if den > 0.0001:
            axis = self.xyz / den
        else:
            axis = Vector3d(1, 0, 0)
        return angle * self.radToDeg, axis

    @staticmethod
    def FromToRotation(fromDirection, toDirection):
        return QuaternionD.RotateTowards(QuaternionD.LookRotation(fromDirection), QuaternionD.LookRotation(toDirection), float('inf'))

    def SetFromToRotation(self, fromDirection, toDirection):
        self = QuaternionD.FromToRotation(fromDirection, toDirection)

    @staticmethod
    def LookRotation(forward, upwards=Vector3d.up()):
        forward = Vector3d.Normalize(forward)
        right = Vector3d.Normalize(Vector3d.Cross(upwards, forward))
        upwards = Vector3d.Cross(forward, right)
        m00, m01, m02 = right.x, right.y, right.z
        m10, m11, m12 = upwards.x, upwards.y, upwards.z
        m20, m21, m22 = forward.x, forward.y, forward.z
        num8 = m00 + m11 + m22
        if num8 > 0.0:
            num = math.sqrt(num8 + 1.0)
            return QuaternionD((m12 - m21) * (0.5 / num), (m20 - m02) * (0.5 / num), (m01 - m10) * (0.5 / num), num * 0.5)
        if m00 >= m11 and m00 >= m22:
            num7 = math.sqrt(((1.0 + m00) - m11) - m22)
            return QuaternionD(0.5 * num7, (m01 + m10) * (0.5 / num7), (m02 + m20) * (0.5 / num7), (m12 - m21) * (0.5 / num7))
        if m11 > m22:
            num6 = math.sqrt(((1.0 + m11) - m00) - m22)
            return QuaternionD((m10 + m01) * (0.5 / num6), 0.5 * num6, (m21 + m12) * (0.5 / num6), (m20 - m02) * (0.5 / num6))
        num5 = math.sqrt(((1.0 + m22) - m00) - m11)
        return QuaternionD((m20 + m02) * (0.5 / num5), (m21 + m12) * (0.5 / num5), 0.5 * num5, (m01 - m10) * (0.5 / num5))

    def SetLookRotation(self, view, up=Vector3d.up()):
        self = QuaternionD.LookRotation(view, up)

    @staticmethod
    def Slerp(a, b, t):
        t = max(0, min(1, t))
        return QuaternionD.SlerpUnclamped(a, b, t)

    @staticmethod
    def SlerpUnclamped(a, b, t):
        if a.LengthSquared == 0.0:
            if b.LengthSquared == 0.0:
                return QuaternionD.identity()
            return b
        elif b.LengthSquared == 0.0:
            return a
        cosHalfAngle = a.w * b.w + Vector3d.Dot(a.xyz, b.xyz)
        if cosHalfAngle >= 1.0 or cosHalfAngle <= -1.0:
            return a
        elif cosHalfAngle < 0.0:
            b.xyz = -b.xyz
            b.w = -b.w
            cosHalfAngle = -cosHalfAngle
        if cosHalfAngle < 0.99:
            halfAngle = math.acos(cosHalfAngle)
            sinHalfAngle = math.sin(halfAngle)
            oneOverSinHalfAngle = 1.0 / sinHalfAngle
            blendA = math.sin(halfAngle * (1.0 - t)) * oneOverSinHalfAngle
            blendB = math.sin(halfAngle * t) * oneOverSinHalfAngle
        else:
            blendA = 1.0 - t
            blendB = t
        result = QuaternionD(blendA * a.xyz + blendB * b.xyz, blendA * a.w + blendB * b.w)
        if result.LengthSquared > 0.0:
            return QuaternionD.Normalize(result)
        else:
            return QuaternionD.identity()

    @staticmethod
    def Lerp(a, b, t):
        t = max(0, min(1, t))
        return QuaternionD.Slerp(a, b, t)

    @staticmethod
    def LerpUnclamped(a, b, t):
        return QuaternionD.Slerp(a, b, t)

    @staticmethod
    def RotateTowards(from_, to, maxDegreesDelta):
        num = QuaternionD.Angle(from_, to)
        if num == 0.0:
            return to
        t = min(1.0, maxDegreesDelta / num)
        return QuaternionD.SlerpUnclamped(from_, to, t)

    @staticmethod
    def Inverse(rotation):
        lengthSq = rotation.LengthSquared
        if lengthSq != 0.0:
            i = 1.0 / lengthSq
            return QuaternionD(rotation.xyz * -i, rotation.w * i)
        return rotation

    def __str__(self):
        return f"({self.x:.1f}, {self.y:.1f}, {self.z:.1f}, {self.w:.1f})"

    def ToString(self, format):
        return f"({format(self.x, format)}, {format(self.y, format)}, {format(self.z, format)}, {format(self.w, format)})"

    @staticmethod
    def Angle(a, b):
        f = QuaternionD.Dot(a, b)
        return math.acos(min(abs(f), 1.0)) * 2.0 * QuaternionD.radToDeg

    @staticmethod
    def Euler(x, y, z):
        return QuaternionD.FromEulerRad(Vector3d(x, y, z) * QuaternionD.degToRad)

    @staticmethod
    def Euler(euler):
        return QuaternionD.FromEulerRad(euler * QuaternionD.degToRad)

    @staticmethod
    def ToEulerRad(rotation):
        sqw = rotation.w * rotation.w
        sqx = rotation.x * rotation.x
        sqy = rotation.y * rotation.y
        sqz = rotation.z * rotation.z
        unit = sqx + sqy + sqz + sqw
        test = rotation.x * rotation.w - rotation.y * rotation.z
        if test > 0.4995 * unit:
            return QuaternionD.NormalizeAngles(Vector3d(2 * math.atan2(rotation.y, rotation.x), math.pi / 2, 0) * QuaternionD.radToDeg)
        if test < -0.4995 * unit:
            return QuaternionD.NormalizeAngles(Vector3d(-2 * math.atan2(rotation.y, rotation.x), -math.pi / 2, 0) * QuaternionD.radToDeg)
        q = QuaternionD(rotation.w, rotation.z, rotation.x, rotation.y)
        v = Vector3d(
            math.atan2(2 * q.x * q.w + 2 * q.y * q.z, 1 - 2 * (q.z * q.z + q.w * q.w)),
            math.asin(2 * (q.x * q.z - q.w * q.y)),
            math.atan2(2 * q.x * q.y + 2 * q.z * q.w, 1 - 2 * (q.y * q.y + q.z * q.z))
        )
        return QuaternionD.NormalizeAngles(v * QuaternionD.radToDeg)

    @staticmethod
    def NormalizeAngles(angles):
        angles.x = QuaternionD.NormalizeAngle(angles.x)
        angles.y = QuaternionD.NormalizeAngle(angles.y)
        angles.z = QuaternionD.NormalizeAngle(angles.z)
        return angles

    @staticmethod
    def NormalizeAngle(angle):
        while angle > 360:
            angle -= 360
        while angle < 0:
            angle += 360
        return angle

    @staticmethod
    def FromEulerRad(euler):
        yaw, pitch, roll = euler.x, euler.y, euler.z
        rollOver2 = roll * 0.5
        sinRollOver2 = math.sin(rollOver2)
        cosRollOver2 = math.cos(rollOver2)
        pitchOver2 = pitch * 0.5
        sinPitchOver2 = math.sin(pitchOver2)
        cosPitchOver2 = math.cos(pitchOver2)
        yawOver2 = yaw * 0.5
        sinYawOver2 = math.sin(yawOver2)
        cosYawOver2 = math.cos(yawOver2)
        return QuaternionD(
            sinYawOver2 * cosPitchOver2 * cosRollOver2 + cosYawOver2 * sinPitchOver2 * sinRollOver2,
            cosYawOver2 * sinPitchOver2 * cosRollOver2 - sinYawOver2 * cosPitchOver2 * sinRollOver2,
            cosYawOver2 * cosPitchOver2 * sinRollOver2 - sinYawOver2 * sinPitchOver2 * cosRollOver2,
            cosYawOver2 * cosPitchOver2 * cosRollOver2 + sinYawOver2 * sinPitchOver2 * sinRollOver2
        )

    @staticmethod
    def ToAxisAngleRad(q):
        if abs(q.w) > 1.0:
            q.Normalize()
        angle = 2.0 * math.acos(q.w)
        den = math.sqrt(1.0 - q.w * q.w)
        if den > 0.0001:
            axis = q.xyz / den
        else:
            axis = Vector3d(1, 0, 0)
        return axis, angle

    def __eq__(self, other):
        return isinstance(other, QuaternionD) and self.x == other.x and self.y == other.y and self.z == other.z and self.w == other.w

    def __hash__(self):
        return hash((self.x, self.y, self.z, self.w))

    def __mul__(self, other):
        if isinstance(other, QuaternionD):
            return QuaternionD(
                self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
                self.w * other.y + self.y * other.w + self.z * other.x - self.x * other.z,
                self.w * other.z + self.z * other.w + self.x * other.y - self.y * other.x,
                self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
            )
        elif isinstance(other, Vector3d):
            num = self.x * 2.0
            num2 = self.y * 2.0
            num3 = self.z * 2.0
            num4 = self.x * num
            num5 = self.y * num2
            num6 = self.z * num3
            num7 = self.x * num2
            num8 = self.x * num3
            num9 = self.y * num3
            num10 = self.w * num
            num11 = self.w * num2
            num12 = self.w * num3
            return Vector3d(
                (1.0 - (num5 + num6)) * other.x + (num7 - num12) * other.y + (num8 + num11) * other.z,
                (num7 + num12) * other.x + (1.0 - (num4 + num6)) * other.y + (num9 - num10) * other.z,
                (num8 - num11) * other.x + (num9 + num10) * other.y + (1.0 - (num4 + num5)) * other.z
            )
        else:
            raise TypeError("Unsupported multiplication")

    def __ne__(self, other):
        return not self.__eq__(other)

def to_list(q):
    """
    Converts a QuaternionD to a list representation.
    """
    return [float(q.x), float(q.y), float(q.z), float(q.w)]

def from_list(q_list):
    """
    Converts a list to a QuaternionD.
    """
    return QuaternionD(float(q_list[0]), float(q_list[1]), float(q_list[2]), float(q_list[3]))

# Example usage:
q1 = QuaternionD(1, 0, 0, 0)
q2 = QuaternionD(0, 1, 0, 0)
q3 = q1 * q2
print(q3)

q_list = to_list(q3)
print(q_list)

q4 = from_list(q_list)
print(q4)

class QuaternionD:
    def __init__(self, x=0, y=0, z=0, w=1):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __repr__(self):
        return f"QuaternionD({self.x}, {self.y}, {self.z}, {self.w})"

    def __mul__(self, other):
        if isinstance(other, QuaternionD):
            return QuaternionD(
                self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
                self.w * other.y + self.y * other.w + self.z * other.x - self.x * other.z,
                self.w * other.z + self.z * other.w + self.x * other.y - self.y * other.x,
                self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            )
        elif isinstance(other, (list, tuple)) and len(other) == 3:
            x, y, z = other
            num = self.x * 2.0
            num2 = self.y * 2.0
            num3 = self.z * 2.0
            num4 = self.x * num
            num5 = self.y * num2
            num6 = self.z * num3
            num7 = self.x * num2
            num8 = self.x * num3
            num9 = self.y * num3
            num10 = self.w * num
            num11 = self.w * num2
            num12 = self.w * num3
            return [
                (1.0 - (num5 + num6)) * x + (num7 - num12) * y + (num8 + num11) * z,
                (num7 + num12) * x + (1.0 - (num4 + num6)) * y + (num9 - num10) * z,
                (num8 - num11) * x + (num9 + num10) * y + (1.0 - (num4 + num5)) * z,
            ]

    @staticmethod
    def from_euler_angles(x, y, z):
        yaw, pitch, roll = x, y, z
        roll_over2 = roll * 0.5
        sin_roll_over2 = math.sin(roll_over2)
        cos_roll_over2 = math.cos(roll_over2)
        pitch_over2 = pitch * 0.5
        sin_pitch_over2 = math.sin(pitch_over2)
        cos_pitch_over2 = math.cos(pitch_over2)
        yaw_over2 = yaw * 0.5
        sin_yaw_over2 = math.sin(yaw_over2)
        cos_yaw_over2 = math.cos(yaw_over2)
        return QuaternionD(
            sin_yaw_over2 * cos_pitch_over2 * cos_roll_over2 + cos_yaw_over2 * sin_pitch_over2 * sin_roll_over2,
            cos_yaw_over2 * sin_pitch_over2 * cos_roll_over2 - sin_yaw_over2 * cos_pitch_over2 * sin_roll_over2,
            cos_yaw_over2 * cos_pitch_over2 * sin_roll_over2 - sin_yaw_over2 * sin_pitch_over2 * cos_roll_over2,
            cos_yaw_over2 * cos_pitch_over2 * cos_roll_over2 + sin_yaw_over2 * sin_pitch_over2 * sin_roll_over2,
        )

    def to_euler_angles(self):
        sqw = self.w * self.w
        sqx = self.x * self.x
        sqy = self.y * self.y
        sqz = self.z * self.z
        unit = sqx + sqy + sqz + sqw
        test = self.x * self.w - self.y * self.z
        if test > 0.4995 * unit:
            return [2 * math.atan2(self.y, self.x), math.pi / 2, 0]
        if test < -0.4995 * unit:
            return [-2 * math.atan2(self.y, self.x), -math.pi / 2, 0]
        return [
            math.atan2(2 * (self.w * self.x + self.y * self.z), 1 - 2 * (self.x * self.x + self.y * self.y)),
            math.asin(2 * (self.w * self.y - self.z * self.x)),
            math.atan2(2 * (self.w * self.z + self.x * self.y), 1 - 2 * (self.y * self.y + self.z * self.z)),
        ]

    @staticmethod
    def slerp(q1, q2, t):
        cos_half_theta = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z
        if abs(cos_half_theta) >= 1.0:
            return q1
        half_theta = math.acos(cos_half_theta)
        sin_half_theta = math.sqrt(1.0 - cos_half_theta * cos_half_theta)
        if abs(sin_half_theta) < 0.001:
            return QuaternionD(
                q1.x * 0.5 + q2.x * 0.5,
                q1.y * 0.5 + q2.y * 0.5,
                q1.z * 0.5 + q2.z * 0.5,
                q1.w * 0.5 + q2.w * 0.5,
            )
        ratio_a = math.sin((1 - t) * half_theta) / sin_half_theta
        ratio_b = math.sin(t * half_theta) / sin_half_theta
        return QuaternionD(
            q1.x * ratio_a + q2.x * ratio_b,
            q1.y * ratio_a + q2.y * ratio_b,
            q1.z * ratio_a + q2.z * ratio_b,
            q1.w * ratio_a + q2.w * ratio_b,
        )

# Test the class with some example operations
q1 = QuaternionD(0, 0, 0, 1)
q2 = QuaternionD(1, 0, 0, 0)

print("q1:", q1)
print("q2:", q2)

# Multiply two quaternions
q3 = q1 * q2
print("q1 * q2:", q3)

# Quaternion from Euler angles
q4 = QuaternionD.from_euler_angles(math.radians(90), 0, 0)
print("Quaternion from Euler angles (90, 0, 0):", q4)

# Euler angles from quaternion
euler = q4.to_euler_angles()
print("Euler angles from quaternion:", euler)

# Slerp between two quaternions
q5 = QuaternionD.slerp(q1, q2, 0.5)
print("Slerp between q1 and q2 at t=0.5:", q5)

class PlanetEditor:
    class Tab:
        General, Generation, Visual, Foliage, Events, Debug = range(6)

    def __init__(self, planet):
        self.tab = self.Tab.General
        self.planet = planet
        self.show_color_editor = False
        self.show_exp_foliage = False
        self.tab_names = ["General", "Generation", "Visual", "Foliage", "Events", "Debug"]
        self.default_sequence = [
            (31, 119, 180, 255),
            (255, 127, 14, 255),
            (44, 160, 44, 255),
            (214, 39, 40, 255),
            (148, 103, 189, 255),
            (140, 86, 75, 255)
        ]

    def on_enable(self):
        self.detail_distances = self.planet.detail_distances
        self.generate_colliders = self.planet.generate_colliders
        self.detail_msds = self.planet.detail_msds
        self.scaled_space_material = self.planet.scaled_space_material
        self.planet_material = self.planet.planet_material
        self.foliage_biomes = self.planet.foliage_biomes
        self.event_sc_sp_entered = self.planet.event_sc_sp_entered
        self.event_sc_sp_left = self.planet.event_sc_sp_left
        self.event_finished_generation = self.planet.event_finished_generation

    def on_inspector_gui(self):
        self.tab = self._draw_toolbar(self.tab, self.tab_names)
        
        if self.tab == self.Tab.General:
            self._draw_general_tab()
        elif self.tab == self.Tab.Generation:
            self._draw_generation_tab()
        elif self.tab == self.Tab.Visual:
            self._draw_visual_tab()
        elif self.tab == self.Tab.Foliage:
            self._draw_foliage_tab()
        elif self.tab == self.Tab.Events:
            self._draw_events_tab()
        elif self.tab == self.Tab.Debug:
            self._draw_debug_tab()

    def _draw_toolbar(self, selected_tab, tab_names):
        # Replace this with actual GUI drawing code
        return selected_tab

    def _draw_general_tab(self):
        self.planet.radius = self._input_float("Radius", self.planet.radius)
        self.planet.calculate_msds = self._input_bool("Calculate MSDs", self.planet.calculate_msds)
        if self.planet.calculate_msds:
            self.detail_msds = self._input_list("Detail MSDs", self.detail_msds)
        self.generate_colliders = self._input_bool("Generate Colliders", self.generate_colliders)
        self.planet.lod_mode_behind_cam = self._input_enum("LOD Mode behind Camera", self.planet.lod_mode_behind_cam)
        if self.planet.lod_mode_behind_cam == "NotComputed":
            self.planet.behind_camera_extra_range = self._input_float("LOD Extra Range", self.planet.behind_camera_extra_range)
        self.planet.recompute_quad_distances_threshold = self._input_float("Recompute Quad Threshold", self.planet.recompute_quad_distances_threshold)
        self.planet.update_all_quads = self._input_bool("Update all Quads simultaneously", self.planet.update_all_quads)
        if not self.planet.update_all_quads:
            self.planet.max_quads_to_update = self._input_int("Max Quads to update per frame", self.planet.max_quads_to_update)
        self.planet.floating_origin = self._input_object("Floating Origin (if used)", self.planet.floating_origin)
        self.planet.hide_quads = self._input_bool("Hide Quads in Hierarchy", self.planet.hide_quads)
        self.planet.quad_size = self._input_int("Quad Size", self.planet.quad_size, 5, 253)

    def _draw_generation_tab(self):
        self.planet.height_provider_type = self._input_enum("Generation Mode", self.planet.height_provider_type)
        if self.planet.height_provider_type == "Heightmap":
            self.planet.heightmap_text_asset = self._input_object("Heightmap", self.planet.heightmap_text_asset)
            self.planet.use_bicubic_interpolation = self._input_bool("Use Bicubic Interpolation", self.planet.use_bicubic_interpolation)
        elif self.planet.height_provider_type == "Noise":
            self.planet.noise_serialized = self._input_object("Noise", self.planet.noise_serialized)
        elif self.planet.height_provider_type == "Hybrid":
            self.planet.heightmap_text_asset = self._input_object("Heightmap", self.planet.heightmap_text_asset)
            self.planet.use_bicubic_interpolation = self._input_bool("Use Bicubic Interpolation", self.planet.use_bicubic_interpolation)
            self.planet.noise_serialized = self._input_object("Noise", self.planet.noise_serialized)
            self.planet.hybrid_mode_noise_div = self._input_float("Noise Divisor", self.planet.hybrid_mode_noise_div)
        elif self.planet.height_provider_type == "Const":
            self.planet.constant_height = self._input_float("Constant Height", self.planet.constant_height)
        elif self.planet.height_provider_type == "ComputeShader":
            self.planet.compute_shader = self._input_object("Compute Shader", self.planet.compute_shader)
        elif self.planet.height_provider_type == "StreamingHeightmap":
            self._draw_streaming_heightmap_tab()

        self.planet.height_scale = self._input_float("Height Scale", self.planet.height_scale)
        self.planet.quads_splitting_simultaneously = self._input_int("Quads Splitting Simultaneously", self.planet.quads_splitting_simultaneously)
        self.planet.use_scaled_space = self._input_bool("Use Scaled Space", self.planet.use_scaled_space)
        if self.planet.use_scaled_space:
            self.planet.scaled_space_factor = self._input_float("Scaled Space Factor", self.planet.scaled_space_factor)

    def _draw_visual_tab(self):
        self.planet_material = self._input_object("Planet Material", self.planet_material)
        self.planet.uv_type = self._input_enum("UV Type", self.planet.uv_type)
        if self.planet.uv_type == "Cube":
            self.planet.uv_scale = self._input_float("UV Scale", self.planet.uv_scale)
        if self.planet.use_scaled_space:
            self.planet.scaled_space_distance = self._input_float("Scaled Space Distance", self.planet.scaled_space_distance)
            self.scaled_space_material = self._input_object("Scaled Space Material", self.scaled_space_material)
        self.planet.vis_sphere_radius_mod = self._input_float("Visibility Sphere Radius Mod", self.planet.vis_sphere_radius_mod)

        self.planet.slope_texture_type = self._input_enum("Slope Texture Type", self.planet.slope_texture_type)
        if self.planet.slope_texture_type == "Threshold":
            self.planet.slope_angle = self._input_float("Slope Angle", self.planet.slope_angle)
            self.planet.slope_texture = self._input_int("Slope Texture", self.planet.slope_texture)
        elif self.planet.slope_texture_type == "Fade":
            self.planet.slope_angle = self._input_float("Slope Angle", self.planet.slope_angle)
            self.planet.slope_fade_in_angle = self._input_float("Fade-in Angle", self.planet.slope_fade_in_angle)
            self.planet.slope_texture = self._input_int("Slope Texture", self.planet.slope_texture)

        self.planet.texture_provider_type = self._input_enum("Texture Selection Type", self.planet.texture_provider_type)
        if self.planet.texture_provider_type == "Gradient":
            self._draw_tex_provider_gradient()
        elif self.planet.texture_provider_type == "Range":
            self._draw_tex_provider_range()
        elif self.planet.texture_provider_type == "Splatmap":
            self._draw_tex_provider_splatmap()

    def _draw_foliage_tab(self):
        self.planet.generate_details = self._input_bool("Generate Details", self.planet.generate_details)
        if self.planet.generate_details:
            self.planet.foliage_biomes = self._input_enum_flags("Foliage Biomes", self.planet.foliage_biomes)
            self.planet.generate_grass = self._input_bool("Generate Grass", self.planet.generate_grass)
            if self.planet.generate_grass:
                self.planet.grass_per_quad = max(0, self._input_int("Grass per Quad", self.planet.grass_per_quad))
                self.planet.grass_material = self._input_object("Grass Material", self.planet.grass_material)
            self.planet.grass_level = self._input_int("Detail Level", self.planet.grass_level)
            self.planet.detail_distance = self._input_float("Detail Distance", self.planet.detail_distance)
            self.planet.detail_objects_generating_simultaneously = self._input_int("Details generating simultaneously", self.planet.detail_objects_generating_simultaneously)

            if self._button("Add Mesh"):
                self.planet.detail_meshes.append(DetailMesh())
            if self._button("Add Prefab"):
                self.planet.detail_prefabs.append(DetailPrefab())

            for i, dM in enumerate(self.planet.detail_meshes):
                if dM.is_grass:
                    continue
                self._draw_detail_mesh(i, dM)

            for i, dP in enumerate(self.planet.detail_prefabs):
                self._draw_detail_prefab(i, dP)

            self.show_exp_foliage = self._foldout(self.show_exp_foliage, "Experimental")
            if self.show_exp_foliage:
                self.planet.exp_grass = self._input_bool("Use exp. backend for Grass", self.planet.exp_grass)
                self.planet.exp_meshes = self._input_bool("Use exp. backend for Meshes", self.planet.exp_meshes)
                self.planet.exp_prefabs = self._input_bool("Use exp. backend for Prefabs", self.planet.exp_prefabs)

    def _draw_events_tab(self):
        self.event_finished_generation = self._input_object("Finished generating Quads", self.event_finished_generation)
        self.event_sc_sp_entered = self._input_object("Player entered Scaled Space", self.event_sc_sp_entered)
        self.event_sc_sp_left = self._input_object("Player left Scaled Space", self.event_sc_sp_left)

    def _draw_debug_tab(self):
        # Draw default inspector or custom debug tools
        pass

    def _draw_streaming_heightmap_tab(self):
        self.planet.base_heightmap_text_asset = self._input_object("Base Heightmap", self.planet.base_heightmap_text_asset)
        self.planet.heightmap_path = self._input_text("Path", self.planet.heightmap_path)
        self.planet.use_bicubic_interpolation = self._input_bool("Use Bicubic Interpolation", self.planet.use_bicubic_interpolation)
        self.planet.load_size = self._input_vector2("Loaded Area Size", self.planet.load_size)
        self.planet.reload_threshold = self._input_float("Reload Threshold", self.planet.reload_threshold)

    def _draw_tex_provider_splatmap(self):
        # Implement splatmap texture provider UI here
        pass

    def _draw_tex_provider_gradient(self):
        # Implement gradient texture provider UI here
        pass

    def _draw_tex_provider_range(self):
        # Implement range texture provider UI here
        pass

    def _draw_detail_mesh(self, index, detail_mesh):
        # Implement detail mesh UI drawing here
        pass

    def _draw_detail_prefab(self, index, detail_prefab):
        # Implement detail prefab UI drawing here
        pass

    def _input_float(self, label, value):
        # Replace this with actual input code
        return value

    def _input_int(self, label, value, min_value=None, max_value=None):
        # Replace this with actual input code
        return value

    def _input_bool(self, label, value):
        # Replace this with actual input code
        return value

    def _input_enum(self, label, value):
        # Replace this with actual input code
        return value

    def _input_enum_flags(self, label, value):
        # Replace this with actual input code
        return value

    def _input_list(self, label, value):
        # Replace this with actual input code
        return value

    def _input_object(self, label, value):
        # Replace this with actual input code
        return value

    def _input_text(self, label, value):
        # Replace this with actual input code
        return value

    def _input_vector2(self, label, value):
        # Replace this with actual input code
        return value

    def _button(self, label):
        # Replace this with actual button code
        return False

    def _foldout(self, state, label):
        # Replace this with actual foldout code
        return state

import numpy as np
from math import radians, sin, cos, sqrt
from typing import List, Tuple, Dict

# Define Enums
class LODModeBehindCam:
    NotComputed, ComputeRender = range(2)

class HeightProviderType:
    Heightmap, Noise, Hybrid, Const, ComputeShader, StreamingHeightmap = range(6)

class UVType:
    Cube, Quad, Legacy, LegacyContinuous = range(4)

class TextureProviderType:
    Gradient, Range, Splatmap, NoneType = range(4)  # Renamed 'None' to 'NoneType'

class SlopeTextureType:
    Fade, Threshold, NoneType = range(3)  # Renamed 'None' to 'NoneType'

# Placeholder Classes for Materials and Events
class Material:
    pass

class UnityEvent:
    def __init__(self):
        self.listeners = []
    
    def invoke(self):
        for listener in self.listeners:
            listener()

    def add_listener(self, listener):
        self.listeners.append(listener)

class GameObject:
    pass

# Placeholder for the QuaternionD class
class QuaternionD:
    def __init__(self, x=0, y=0, z=0, w=1):
        self.x, self.y, self.z, self.w = x, y, z, w

    @staticmethod
    def identity():
        return QuaternionD(0, 0, 0, 1)

    def to_rotation_matrix(self):
        x2, y2, z2 = self.x * self.x, self.y * self.y, self.z * self.z
        xy, xz, yz = self.x * self.y, self.x * self.z, self.y * self.z
        wx, wy, wz = self.w * self.x, self.w * self.y, self.w * self.z

        return np.array([
            [1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2)]
        ])

class DetailMesh:
    pass

class DetailPrefab:
    pass

class FloatingOrigin:
    pass

class IHeightProvider:
    def init(self):
        pass

class ITextureProvider:
    pass

# Helper functions
def lat_lon_to_xyz(lat_lon, radius):
    lat, lon = radians(lat_lon[0]), radians(lat_lon[1])
    x = radius * cos(lat) * cos(lon)
    y = radius * cos(lat) * sin(lon)
    z = radius * sin(lat)
    return np.array([x, y, z])

class Planet:
    def __init__(self):
        self.radius = 10000
        self.rotation = QuaternionD.identity()
        self.detail_distances = [50000, 25000, 12500, 6250, 3125]
        self.calculate_msds = False
        self.detail_msds = [0, 0, 0, 0, 0]
        self.lod_mode_behind_cam = LODModeBehindCam.ComputeRender
        self.behind_camera_extra_range = 0.0
        self.planet_material = Material()
        self.uv_type = UVType.Cube
        self.uv_scale = 1.0
        self.generate_colliders = [False, False, False, False, False, True]
        self.vis_sphere_radius_mod = 1.0
        self.update_all_quads = False
        self.max_quads_to_update = 250
        self.recompute_quad_distances_threshold = 10.0
        self.quads_splitting_simultaneously = 2
        self.quad_size = 33

        self.use_scaled_space = False
        self.scaled_space_distance = 1500.0
        self.scaled_space_factor = 100000.0
        self.scaled_space_material = Material()
        self.scaled_space_copy = None

        self.texture_provider = ITextureProvider()
        self.slope_texture_type = SlopeTextureType.NoneType  # Renamed 'None' to 'NoneType'
        self.slope_fade_in_angle = 10
        self.slope_angle = 60
        self.slope_texture = 5

        self.height_provider = IHeightProvider()
        self.compute_shader = None
        self.height_scale = 0.02

        self.generate_details = False
        self.generate_grass = False
        self.grass_material = Material()
        self.grass_per_quad = 10000
        self.grass_level = 5
        self.detail_distance = 0.0
        self.exp_grass = False
        self.exp_meshes = False
        self.exp_prefabs = False
        self.detail_meshes = []
        self.detail_prefabs = []
        self.foliage_biomes = None
        self.detail_objects_generating_simultaneously = 3

        self.floating_origin = FloatingOrigin()
        self.hide_quads = True

        self.num_quads = 0
        self.texture_color_sequence = [
            (31, 119, 180, 255),
            (255, 127, 14, 255),
            (44, 160, 44, 255),
            (214, 39, 40, 255),
            (148, 103, 189, 255),
            (140, 86, 75, 255)
        ]

        self.quad_split_queue = None
        self.quad_pool = None
        self.quad_game_object_pool = None
        self.quad_arrays = None
        self.quads = []
        self.quad_indices = {}
        self.view_planes = None
        self.initialized = False
        self.in_scaled_space = False
        self.using_legacy_uv_type = False
        self.detail_distance_sqr = 0.0
        self.height_inv = 0.0
        self.radius_vis_sphere = 0.0
        self.detail_distances_sqr = []
        self.detail_objects_generating = 0
        self.world_to_mesh_vector = np.zeros(3)
        self.quad_go = None

        self.main_camera = None
        self.main_camera_tr = None

        self.radius_sqr = 0.0
        self.radius_max_sqr = 0.0
        self.scaled_space_dis_sqr = 0.0
        self.recompute_quad_distances_threshold_sqr = 0.0

        self.old_cam_rotation = QuaternionD.identity()
        self.old_cam_position = np.zeros(3)
        self.old_planet_position = np.zeros(3)
        self.old_planet_rotation = QuaternionD.identity()
        self.quad_update_cv = None

        self.frames_since_split = 0

        self.scaled_space_state_changed = None
        self.entered_scaled_space = UnityEvent()
        self.left_scaled_space = UnityEvent()
        self.finished_generation = None
        self.event_finished_generation = UnityEvent()

        self.serialized_inherited = self.SerializedInheritedClasses()

    class SerializedInheritedClasses:
        def __init__(self):
            self.height_provider_type = HeightProviderType.Const
            self.heightmap_height_provider = None
            self.noise_height_provider = None
            self.hybrid_height_provider = None
            self.const_height_provider = None
            self.streaming_heightmap_height_provider = None

            self.texture_provider_type = TextureProviderType.NoneType  # Renamed 'None' to 'NoneType'
            self.texture_provider_gradient = None
            self.texture_provider_range = None
            self.texture_provider_splatmap = None


    def initialize(self):
        # Initialize height provider
        if self.serialized_inherited.height_provider_type == HeightProviderType.Heightmap:
            self.height_provider = self.serialized_inherited.heightmap_height_provider
        elif self.serialized_inherited.height_provider_type == HeightProviderType.Noise:
            self.height_provider = self.serialized_inherited.noise_height_provider
        elif self.serialized_inherited.height_provider_type == HeightProviderType.Hybrid:
            self.height_provider = self.serialized_inherited.hybrid_height_provider
        elif self.serialized_inherited.height_provider_type == HeightProviderType.StreamingHeightmap:
            self.height_provider = self.serialized_inherited.streaming_heightmap_height_provider
        else:
            self.height_provider = self.serialized_inherited.const_height_provider

        self.height_provider.init()

        # Initialize texture provider
        if self.serialized_inherited.texture_provider_type == TextureProviderType.Gradient:
            self.texture_provider = self.serialized_inherited.texture_provider_gradient
        elif self.serialized_inherited.texture_provider_type == TextureProviderType.Range:
            self.texture_provider = self.serialized_inherited.texture_provider_range
        elif self.serialized_inherited.texture_provider_type == TextureProviderType.Splatmap:
            self.texture_provider = self.serialized_inherited.texture_provider_splatmap
            self.texture_provider.init()
        else:
            self.texture_provider = ITextureProvider()

        self.using_legacy_uv_type = self.uv_type in [UVType.Legacy, UVType.LegacyContinuous]

        self.height_inv = 1.0 / self.height_scale
        self.rotation = self.transform_rotation_to_quaterniond()
        self.old_planet_position = self.transform_position()
        self.old_planet_rotation = self.rotation

        self.radius_max_sqr = self.radius * (self.height_inv + 1) / self.height_inv
        self.radius_max_sqr *= self.radius_max_sqr
        self.radius_sqr = self.radius * self.radius
        self.recompute_quad_distances_threshold_sqr = self.recompute_quad_distances_threshold ** 2
        self.scaled_space_dis_sqr = self.scaled_space_distance ** 2
        self.detail_distance_sqr = self.detail_distance ** 2

        self.detail_distances_sqr = [d * d for d in self.detail_distances]

        self.quad_split_queue = []
        self.quad_pool = []
        self.quad_game_object_pool = []

        self.initialize_scaled_space()

    def initialize_scaled_space(self):
        cam_height = np.linalg.norm(self.main_camera_tr_position() - self.transform_position())
        if self.use_scaled_space:
            if cam_height > self.scaled_space_dis_sqr:
                self.in_scaled_space = True
                self.entered_scaled_space.invoke()
            elif cam_height < self.scaled_space_dis_sqr:
                self.in_scaled_space = False
                self.left_scaled_space.invoke()

    def update(self):
        self.transform_set_rotation(self.rotation)

        cam_pos = self.main_camera_tr_position()
        cam_rot = self.main_camera_tr_rotation()

        tr_pos = self.transform_position()
        tr_rot = self.transform_rotation()

        rel_cam_pos = cam_pos - tr_pos

        if isinstance(self.height_provider, StreamingHeightmapHeightProvider):
            self.height_provider.update(self.quad_split_queue, np.linalg.norm(cam_pos - tr_pos))

        # Placeholder for the rest of the update logic
        pass

    # Placeholder methods for Unity-specific functionality
    def transform_position(self):
        return np.zeros(3)

    def transform_rotation(self):
        return QuaternionD.identity()

    def transform_set_rotation(self, rotation):
        pass

    def main_camera_tr_position(self):
        return np.zeros(3)

    def main_camera_tr_rotation(self):
        return QuaternionD.identity()

    def transform_rotation_to_quaterniond(self):
        return QuaternionD.identity()

import numpy as np

class LUMINANCE:
    NONE, APPROXIMATE, PRECOMPUTED = range(3)

class CONSTANTS:
    NUM_THREADS = 8

    TRANSMITTANCE_WIDTH = 256
    TRANSMITTANCE_HEIGHT = 64
    TRANSMITTANCE_CHANNELS = 3
    TRANSMITTANCE_SIZE = TRANSMITTANCE_WIDTH * TRANSMITTANCE_HEIGHT

    SCATTERING_R = 32
    SCATTERING_MU = 128
    SCATTERING_MU_S = 32
    SCATTERING_NU = 8

    SCATTERING_WIDTH = SCATTERING_NU * SCATTERING_MU_S
    SCATTERING_HEIGHT = SCATTERING_MU
    SCATTERING_DEPTH = SCATTERING_R
    SCATTERING_CHANNELS = 4
    SCATTERING_SIZE = SCATTERING_WIDTH * SCATTERING_HEIGHT * SCATTERING_DEPTH

    IRRADIANCE_WIDTH = 64
    IRRADIANCE_HEIGHT = 16
    IRRADIANCE_CHANNELS = 3
    IRRADIANCE_SIZE = IRRADIANCE_WIDTH * IRRADIANCE_HEIGHT

    MAX_LUMINOUS_EFFICACY = 683.0

    CIE_2_DEG_COLOR_MATCHING_FUNCTIONS = np.array([
        [360, 0.000129900000, 0.000003917000, 0.000606100000],
        [365, 0.000232100000, 0.000006965000, 0.001086000000],
        [370, 0.000414900000, 0.000012390000, 0.001946000000],
        [375, 0.000741600000, 0.000022020000, 0.003486000000],
        [380, 0.001368000000, 0.000039000000, 0.006450001000],
        [385, 0.002236000000, 0.000064000000, 0.010549990000],
        [390, 0.004243000000, 0.000120000000, 0.020050010000],
        [395, 0.007650000000, 0.000217000000, 0.036210000000],
        [400, 0.014310000000, 0.000396000000, 0.067850010000],
        [405, 0.023190000000, 0.000640000000, 0.110200000000],
        [410, 0.043510000000, 0.001210000000, 0.207400000000],
        [415, 0.077630000000, 0.002180000000, 0.371300000000],
        [420, 0.134380000000, 0.004000000000, 0.645600000000],
        [425, 0.214770000000, 0.007300000000, 1.039050100000],
        [430, 0.283900000000, 0.011600000000, 1.385600000000],
        [435, 0.328500000000, 0.016840000000, 1.622960000000],
        [440, 0.348280000000, 0.023000000000, 1.747060000000],
        [445, 0.348060000000, 0.029800000000, 1.782600000000],
        [450, 0.336200000000, 0.038000000000, 1.772110000000],
        [455, 0.318700000000, 0.048000000000, 1.744100000000],
        [460, 0.290800000000, 0.060000000000, 1.669200000000],
        [465, 0.251100000000, 0.073900000000, 1.528100000000],
        [470, 0.195360000000, 0.090980000000, 1.287640000000],
        [475, 0.142100000000, 0.112600000000, 1.041900000000],
        [480, 0.095640000000, 0.139020000000, 0.812950100000],
        [485, 0.057950010000, 0.169300000000, 0.616200000000],
        [490, 0.032010000000, 0.208020000000, 0.465180000000],
        [495, 0.014700000000, 0.258600000000, 0.353300000000],
        [500, 0.004900000000, 0.323000000000, 0.272000000000],
        [505, 0.002400000000, 0.407300000000, 0.212300000000],
        [510, 0.009300000000, 0.503000000000, 0.158200000000],
        [515, 0.029100000000, 0.608200000000, 0.111700000000],
        [520, 0.063270000000, 0.710000000000, 0.078249990000],
        [525, 0.109600000000, 0.793200000000, 0.057250010000],
        [530, 0.165500000000, 0.862000000000, 0.042160000000],
        [535, 0.225749900000, 0.914850100000, 0.029840000000],
        [540, 0.290400000000, 0.954000000000, 0.020300000000],
        [545, 0.359700000000, 0.980300000000, 0.013400000000],
        [550, 0.433449900000, 0.994950100000, 0.008749999000],
        [555, 0.512050100000, 1.000000000000, 0.005749999000],
        [560, 0.594500000000, 0.995000000000, 0.003900000000],
        [565, 0.678400000000, 0.978600000000, 0.002749999000],
        [570, 0.762100000000, 0.952000000000, 0.002100000000],
        [575, 0.842500000000, 0.915400000000, 0.001800000000],
        [580, 0.916300000000, 0.870000000000, 0.001650001000],
        [585, 0.978600000000, 0.816300000000, 0.001400000000],
        [590, 1.026300000000, 0.757000000000, 0.001100000000],
        [595, 1.056700000000, 0.694900000000, 0.001000000000],
        [600, 1.062200000000, 0.631000000000, 0.000800000000],
        [605, 1.045600000000, 0.566800000000, 0.000600000000],
        [610, 1.002600000000, 0.503000000000, 0.000340000000],
        [615, 0.938400000000, 0.441200000000, 0.000240000000],
        [620, 0.854449900000, 0.381000000000, 0.000190000000],
        [625, 0.751400000000, 0.321000000000, 0.000100000000],
        [630, 0.642400000000, 0.265000000000, 0.000049999990],
        [635, 0.541900000000, 0.217000000000, 0.000030000000],
        [640, 0.447900000000, 0.175000000000, 0.000020000000],
        [645, 0.360800000000, 0.138200000000, 0.000010000000],
        [650, 0.283500000000, 0.107000000000, 0.000000000000],
        [655, 0.218700000000, 0.081600000000, 0.000000000000],
        [660, 0.164900000000, 0.061000000000, 0.000000000000],
        [665, 0.121200000000, 0.044580000000, 0.000000000000],
        [670, 0.087400000000, 0.032000000000, 0.000000000000],
        [675, 0.063600000000, 0.023200000000, 0.000000000000],
        [680, 0.046770000000, 0.017000000000, 0.000000000000],
        [685, 0.032900000000, 0.011920000000, 0.000000000000],
        [690, 0.022700000000, 0.008210000000, 0.000000000000],
        [695, 0.015840000000, 0.005723000000, 0.000000000000],
        [700, 0.011359160000, 0.004102000000, 0.000000000000],
        [705, 0.008110916000, 0.002929000000, 0.000000000000],
        [710, 0.005790346000, 0.002091000000, 0.000000000000],
        [715, 0.004109457000, 0.001484000000, 0.000000000000],
        [720, 0.002899327000, 0.001047000000, 0.000000000000],
        [725, 0.002049190000, 0.000740000000, 0.000000000000],
        [730, 0.001439971000, 0.000520000000, 0.000000000000],
        [735, 0.000999949300, 0.000361100000, 0.000000000000],
        [740, 0.000690078600, 0.000249200000, 0.000000000000],
        [745, 0.000476021300, 0.000171900000, 0.000000000000],
        [750, 0.000332301100, 0.000120000000, 0.000000000000],
        [755, 0.000234826100, 0.000084800000, 0.000000000000],
        [760, 0.000166150500, 0.000060000000, 0.000000000000],
        [765, 0.000117413000, 0.000042400000, 0.000000000000],
        [770, 0.000083075270, 0.000030000000, 0.000000000000],
        [775, 0.000058706520, 0.000021200000, 0.000000000000],
        [780, 0.000041509940, 0.000014990000, 0.000000000000],
        [785, 0.000029353260, 0.000010600000, 0.000000000000],
        [790, 0.000020673830, 0.000007465700, 0.000000000000],
        [795, 0.000014559770, 0.000005257800, 0.000000000000],
        [800, 0.000010253980, 0.000003702900, 0.000000000000],
        [805, 0.000007221456, 0.000002607800, 0.000000000000],
        [810, 0.000005085868, 0.000001836600, 0.000000000000],
        [815, 0.000003581652, 0.000001293400, 0.000000000000],
        [820, 0.000002522525, 0.000000910930, 0.000000000000],
        [825, 0.000001776509, 0.000000641530, 0.000000000000],
        [830, 0.000001251141, 0.000000451810, 0.000000000000]
    ])

    XYZ_TO_SRGB = np.array([
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570]
    ])

import numpy as np
import struct

class Heightmap:
    def __init__(self, data, use_bicubic_interpolation, from_file=False, is_texture=False, channel=0):
        self.use_bicubic_interpolation = use_bicubic_interpolation
        self.disable_interpolation = False
        if from_file:
            self.read_file_bytes(data)
        elif is_texture:
            self.read_texture(data, channel)
        else:
            self.read_file_bytes(data)

        self.m_index_x = self.width - 1
        self.m_index_y = self.height - 1

    def set_pixel_at_pos(self, pos, value):
        lat = (np.pi - np.arccos(pos[1]))
        lon = (np.arctan2(pos[2], pos[0]) + np.pi)

        lat *= (1 / np.pi)
        lon *= (1 / (2 * np.pi))

        lat = min(lat, 0.9999999)
        lon = min(lon, 0.9999999)

        lat *= self.height
        lon *= self.width

        x = round(lon)
        y = round(lat)

        x = x if x != self.width else 0
        y = y if y != self.height else 0

        if self.is_16bit:
            self.ushorts[y * self.width + x] = int(value * np.iinfo(np.uint16).max)
        else:
            self.bytes[y * self.width + x] = int(value * np.iinfo(np.uint8).max)

    def get_pos_interpolated(self, pos):
        lat = (np.pi - np.arccos(pos[1]))
        lon = (np.arctan2(pos[2], pos[0]) + np.pi)

        lat *= (1 / np.pi)
        lon *= (1 / (2 * np.pi))

        lat = min(lat, 0.9999999)
        lon = min(lon, 0.9999999)

        lat *= self.height
        lon *= self.width

        result = 0.0

        if self.use_bicubic_interpolation and not self.disable_interpolation:
            result = self._bicubic_interpolation(lon, lat)
        elif not self.disable_interpolation:
            result = self._bilinear_interpolation(lon, lat)
        else:
            x = round(lon)
            y = round(lat)

            x = x if x != self.width else 0
            y = y if y != self.height else 0

            result = self._get_pixel(x, y)

        return result

    def _get_pixel(self, x, y):
        if self.is_16bit:
            return self.ushorts[y * self.width + x] / np.iinfo(np.uint16).max
        else:
            return self.bytes[y * self.width + x] / np.iinfo(np.uint8).max

    def _bicubic_interpolation(self, lon, lat):
        x2 = int(lon)
        x1, x3, x4 = (x2 - 1) % self.width, (x2 + 1) % self.width, (x2 + 2) % self.width
        y2 = int(lat)
        y1, y3, y4 = (y2 - 1) % self.height, (y2 + 1) % self.height, (y2 + 2) % self.height

        pixels = self._get_surrounding_pixels(x1, x2, x3, x4, y1, y2, y3, y4)

        xpos = lon - x2

        val1 = self._cubic_interpolation(pixels[0:4], xpos)
        val2 = self._cubic_interpolation(pixels[4:8], xpos)
        val3 = self._cubic_interpolation(pixels[8:12], xpos)
        val4 = self._cubic_interpolation(pixels[12:16], xpos)

        return self._cubic_interpolation([val1, val2, val3, val4], lat - y2)

    def _bilinear_interpolation(self, lon, lat):
        x1 = int(lon)
        x2 = (x1 + 1) % self.width
        y1 = int(lat)
        y2 = (y1 + 1) % self.height

        pixels = self._get_surrounding_pixels(x1, x2, y1, y2)

        xpos = lon - x1

        val1 = pixels[0] + (pixels[1] - pixels[0]) * xpos
        val2 = pixels[2] + (pixels[3] - pixels[2]) * xpos

        return val1 + (val2 - val1) * (lat - y1)

    def _get_surrounding_pixels(self, *coords):
        if self.is_16bit:
            return [self.ushorts[y * self.width + x] for x, y in coords]
        else:
            return [self.bytes[y * self.width + x] for x, y in coords]

    def _cubic_interpolation(self, p, x):
        return p[1] + 0.5 * x * (p[2] - p[0] + x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + x * (3.0 * (p[1] - p[2]) + p[3] - p[0])))

    def read_file_bytes(self, file_bytes):
        self.width = struct.unpack('i', file_bytes[0:4])[0]
        self.height = struct.unpack('i', file_bytes[4:8])[0]
        self.is_16bit = struct.unpack('?', file_bytes[8:9])[0]

        length = self.width * self.height

        if self.is_16bit:
            self.ushorts = np.frombuffer(file_bytes[9:], dtype=np.uint16)
        else:
            self.bytes = np.frombuffer(file_bytes[9:], dtype=np.uint8)

    def read_texture(self, texture, channel=0):
        self.width, self.height = texture.shape[:2]
        if channel == -1:
            self.bytes = np.mean(texture, axis=2).astype(np.uint8)
        else:
            self.bytes = texture[:, :, channel].astype(np.uint8)
        self.ushorts = None
        self.is_16bit = False

    def get_texture(self):
        colors = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        if self.is_16bit:
            grayscale = (self.ushorts / 257).astype(np.uint8)
        else:
            grayscale = self.bytes

        colors[:, :, 0] = grayscale
        colors[:, :, 1] = grayscale
        colors[:, :, 2] = grayscale
        colors[:, :, 3] = np.iinfo(np.uint8).max

        return colors

    def get_file_bytes(self):
        header = struct.pack('ii?', self.width, self.height, self.is_16bit)
        if self.is_16bit:
            data = self.ushorts.tobytes()
        else:
            data = self.bytes.tobytes()
        return header + data

    @staticmethod
    def test_heightmap_resolution(length, width, height, is_16bit):
        expected_length = width * height * (2 if is_16bit else 1) + 9
        if length != expected_length:
            raise ValueError("Heightmap resolution incorrect!")


import tkinter as tk
from tkinter import filedialog, colorchooser, simpledialog
import numpy as np
from PIL import Image, ImageTk
import os

class HeightmapGenerator:
    def __init__(self, master):
        self.master = master
        self.master.title("Heightmap Generator")

        self.filename = "heightmapGenerated"
        self.resolution_x = 8192
        self.resolution_y = 4096
        self.texture_heights = [0.0, 0.01, 0.4, 0.8, 1.0]
        self.colors = [(166, 130, 90, 255), (72, 80, 28, 255), (60, 53, 37, 255), (81, 81, 81, 255), (255, 255, 255, 255)]
        self.texture_ids = [0, 1, 2, 3, 4, 5]
        self.ocean = False
        self.ocean_level = 0.0
        self.ocean_color = (48, 57, 56, 255)
        self.heightmap_tex = None
        self.texture = None
        self.data_source = 'Noise'
        self.color_data_source = 'Planet'
        self.width = self.resolution_x
        self.height = self.resolution_y
        self.heightmap_16bit = False
        self.heightmap = None
        self.progress = 0.0
        self.preview = False

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.master, text="Heightmap/Texture Generator", font=("Helvetica", 16)).pack(pady=10)

        tk.Label(self.master, text="Filename").pack()
        self.filename_entry = tk.Entry(self.master)
        self.filename_entry.pack()
        self.filename_entry.insert(0, self.filename)

        tk.Button(self.master, text="Generate Gradient", command=self.generate_gradient).pack(pady=5)

        self.canvas = tk.Canvas(self.master, width=512, height=32)
        self.canvas.pack()

        tk.Button(self.master, text="Generate Preview", command=self.generate_preview).pack(pady=5)
        tk.Button(self.master, text="Generate", command=self.generate).pack(pady=5)

        tk.Label(self.master, text="Ocean Settings", font=("Helvetica", 12)).pack(pady=10)
        self.ocean_var = tk.IntVar()
        tk.Checkbutton(self.master, text="Generate Ocean", variable=self.ocean_var).pack()
        tk.Label(self.master, text="Water Level").pack()
        self.ocean_level_scale = tk.Scale(self.master, from_=0, to=1, orient=tk.HORIZONTAL, resolution=0.01)
        self.ocean_level_scale.pack()
        self.ocean_color_button = tk.Button(self.master, text="Select Ocean Color", command=self.select_ocean_color)
        self.ocean_color_button.pack(pady=5)

        self.progress_label = tk.Label(self.master, text="")
        self.progress_label.pack(pady=10)

    def generate_gradient(self):
        gradient = np.zeros((32, 512, 4), dtype=np.uint8)
        for x in range(512):
            t = x / 511
            color = self.evaluate_color(t)
            gradient[:, x, :] = color

        self.heightmap_tex = ImageTk.PhotoImage(image=Image.fromarray(gradient))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.heightmap_tex)

    def generate_preview(self):
        self.preview = True
        self.width = 512
        self.height = 256
        self.generate_bytes()
        self.generate_textures()
        self.save_assets()

    def generate(self):
        self.preview = False
        self.width = self.resolution_x
        self.height = self.resolution_y
        self.generate_bytes()
        self.generate_textures()
        self.save_assets()

    def generate_bytes(self):
        self.heightmap = np.zeros((self.height, self.width), dtype=np.float32)
        for x in range(self.width):
            for y in range(self.height):
                self.heightmap[y, x] = self.noise_function(x, y)
            self.progress = (x + 1) / self.width
            self.progress_label.config(text=f"Progress: {self.progress * 100:.2f}%")
            self.master.update()

    def noise_function(self, x, y):
        return np.random.rand()

    def generate_textures(self):
        self.heightmap_tex = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.texture = np.zeros((self.height, self.width, 4), dtype=np.uint8)

        for x in range(self.width):
            for y in range(self.height):
                value = self.heightmap[y, x]
                self.heightmap_tex[y, x] = [value * 255] * 3
                color = self.evaluate_color(value)
                self.texture[y, x] = color

        self.heightmap_tex = Image.fromarray(self.heightmap_tex)
        self.texture = Image.fromarray(self.texture)

    def evaluate_color(self, t):
        idx = np.searchsorted(self.texture_heights, t)
        if idx == 0:
            return self.colors[0]
        elif idx == len(self.texture_heights):
            return self.colors[-1]
        else:
            t0, t1 = self.texture_heights[idx - 1], self.texture_heights[idx]
            c0, c1 = self.colors[idx - 1], self.colors[idx]
            factor = (t - t0) / (t1 - t0)
            return tuple((np.array(c0) * (1 - factor) + np.array(c1) * factor).astype(np.uint8))

    def save_assets(self):
        filename = self.filename_entry.get()
        if self.preview:
            self.heightmap_tex.save(f"{filename}_heightmap_preview.png")
            self.texture.save(f"{filename}_texture_preview.png")
        else:
            self.heightmap_tex.save(f"{filename}_heightmap.png")
            self.texture.save(f"{filename}_texture.png")

    def select_ocean_color(self):
        color = colorchooser.askcolor(initialcolor=self.ocean_color)[0]
        if color:
            self.ocean_color = (*map(int, color), 255)
            self.ocean_color_button.config(bg=f"#{self.ocean_color[0]:02x}{self.ocean_color[1]:02x}{self.ocean_color[2]:02x}")

if __name__ == "__main__":
    root = tk.Tk()
    app = HeightmapGenerator(root)
    root.mainloop()

import numpy as np
from enum import Enum, Flag, auto
from collections import defaultdict
import math

class QuadPlane(Enum):
    XPlane = auto()
    YPlane = auto()
    ZPlane = auto()

class Position(Enum):
    Back = auto()
    Front = auto()

class EdgeConfiguration(Flag):
    NoEdge = 0  # Renamed 'None' to 'NoEdge'
    Right = 0xFF
    Left = 0xFF00
    Down = 0xFF0000
    Up = ~0x00FFFFFF
    All = ~0

class Biome(Flag):
    NoBiome = 0  # Renamed 'None' to 'NoBiome'
    Zero = 1
    One = 2
    Two = 4
    Three = 8
    Four = 16
    Five = 32
    All = 63


class Quad:
    order_of_children = {
        (0, 1): [2, 0, 1, 3],
        (0, 0): [3, 1, 0, 2],
        (1, 1): [1, 0, 2, 3],
        (1, 0): [3, 2, 0, 1],
        (2, 1): [3, 2, 0, 1],
        (2, 0): [2, 3, 1, 0],
    }

    def __init__(self, position, rotation):
        self.planet = None
        self.parent = None
        self.children = None
        self.neighbors = None
        self.configuration = EdgeConfiguration.NoEdge  # Renamed 'None' to 'NoEdge'
        self.level = 0
        self.index = 0
        self.plane = None
        self.position = None
        self.has_split = False
        self.is_splitting = False
        self.in_split_queue = False
        self.initialized = False
        self.mesh = None
        self.tr_position = position
        self.rotation = rotation
        self.rendered_quad = None
        self.mesh_offset = np.zeros(3)
        self.coroutine = None
        self.distance = float('inf')
        self.biome = Biome.NoBiome  # Renamed 'None' to 'NoBiome'
        self.scale = 1.0
        self.msd = 0.0
        self.mesh_generator = None
        self.neighbor_ids = None
        self.foliage_renderer = None
        self.visible_to_camera = False

    def reset(self):
        self.biome = Biome.NoBiome  # Renamed 'None' to 'NoBiome'
        self.parent = None
        self.children = None
        self.neighbors = None
        self.configuration = EdgeConfiguration.NoEdge  # Renamed 'None' to 'NoEdge'
        self.has_split = False
        self.is_splitting = False
        self.initialized = False
        self.neighbor_ids = None
        self.mesh_offset = np.zeros(3)
        self.foliage_renderer = None
        self.distance = float('inf')
        self.coroutine = None
        self.level = 1
        self.msd = 0.0
        self.mesh_generator = None

    def apply_to_mesh(self, md):
        self.mesh = md
        self.initialized = True

    def update_distances(self):
        if self.initialized:
            if self.mesh is not None:
                self.distance = self.calculate_distance()
                self.visible_to_camera = self.visible_to_camera_func()

            if self.level < len(self.planet.detail_distances_sqr):
                if self.distance < self.planet.detail_distances_sqr[self.level] and self.visible_to_camera and (not self.planet.calculate_msds or self.msd >= self.planet.detail_msds[self.level]) and not self.has_split and not self.is_splitting:
                    self.planet.quad_split_queue.append(self)

                if self.distance > self.planet.detail_distances_sqr[self.level] or not self.visible_to_camera:
                    if self.in_split_queue:
                        self.planet.quad_split_queue.remove(self)
                    if self.has_split:
                        self.combine()

            if not self.rendered_quad and self.visible_to_camera and not self.has_split and not self.is_splitting and not self.planet.in_scaled_space:
                self.rendered_quad = True  # Simplified for the sake of the example
            elif self.rendered_quad and (not self.visible_to_camera or self.planet.in_scaled_space or self.has_split):
                self.rendered_quad = False  # Simplified for the sake of the example

            if self.rendered_quad and not self.rendered_quad and (self.level == 0 or self.parent.has_split):
                self.rendered_quad = True
                if self.index not in self.planet.quad_indices:
                    self.planet.quad_indices[self.index] = self
                self.update_neighbors()

            if self.planet.generate_details and ((self.biome & self.planet.foliage_biomes) != Biome.NoBiome or (self.planet.foliage_biomes & Biome.All) == Biome.All):
                if self.level >= self.planet.grass_level and self.foliage_renderer is None and self.rendered_quad and self.distance < self.planet.detail_distance_sqr:
                    self.foliage_renderer = True  # Simplified for the sake of the example
                elif self.foliage_renderer and self.distance > self.planet.detail_distance_sqr:
                    self.foliage_renderer = None


    def update(self):
        if self.mesh_generator and self.mesh_generator.is_running and self.mesh_generator.is_completed:
            self.apply_to_mesh(self.mesh_generator.get_mesh_data())
            self.mesh_generator.dispose()
            self.mesh_generator = None
            self.initialized = True

    def get_neighbors(self):
        if self.neighbor_ids is None:
            self.neighbor_ids = [self.calculate_neighbor_id(i) for i in range(4)]
        self.neighbors = [self.planet.quad_indices.get(nid) for nid in self.neighbor_ids]
        configuration_old = self.configuration
        self.configuration = EdgeConfiguration.NoEdge  # Renamed 'None' to 'NoEdge'


        for i, neighbor in enumerate(self.neighbors):
            if neighbor is not None:
                delta = self.level - neighbor.level
                if delta > 0:
                    delta <<= 24
                    self.configuration |= EdgeConfiguration(delta >> (8 * i))

        if self.configuration != configuration_old and self.initialized and len(self.mesh.vertices) > 0:
            self.mesh.triangles = self.planet.quad_arrays.get_triangles(self.configuration)

    def start_mesh_generation(self):
        if self.planet.height_provider_type == 'ComputeShader':
            self.mesh_generator = GPUMeshGenerator(self.planet, self)
        else:
            self.mesh_generator = CPUMeshGenerator(self.planet, self)
        self.mesh_generator.start_generation()

    def split(self):
        if not self.has_split:
            self.is_splitting = True
            self.children = [None] * 4
            order = self.order_of_children[(self.plane.value, self.position.value)]

            for i in range(4):
                pos = self.calculate_child_position(i, order)
                self.children[order[i]] = Quad(pos, self.rotation)
                self.children[order[i]].initialize(self, i)

            for child in self.children:
                child.start_mesh_generation()
                while not child.initialized:
                    child.update()
                child.update_distances()

            for child in self.children:
                if child.rendered_quad:
                    child.rendered_quad = True
                    self.planet.quad_indices[child.index] = child

            for child in self.children:
                child.get_neighbors()

            if self.rendered_quad:
                self.planet.quad_game_object_pool.remove(self)

            self.update_neighbors()
            self.is_splitting = False
            self.has_split = True

    def combine(self):
        if self.has_split and not self.is_splitting:
            self.has_split = False
            for child in self.children:
                if child.has_split:
                    child.combine()
                self.planet.quad_pool.remove(child)
            self.children = None

    def visible_to_camera_func(self):
        return self.distance <= self.planet.radius_vis_sphere

    def calculate_distance(self):
        # Placeholder for distance calculation
        return np.linalg.norm(self.planet.world_to_mesh_vector - self.mesh_offset)

    def calculate_neighbor_id(self, direction):
        # Placeholder for neighbor ID calculation
        return self.index + direction

    def calculate_child_position(self, i, order):
        scale_half = 0.5 * self.scale
        if self.plane == QuadPlane.XPlane:
            return [self.tr_position[0], self.tr_position[1] + (-1)**(i//2) * scale_half, self.tr_position[2] + (-1)**(i%2) * scale_half]
        elif self.plane == QuadPlane.YPlane:
            return [self.tr_position[0] + (-1)**(i//2) * scale_half, self.tr_position[1], self.tr_position[2] + (-1)**(i%2) * scale_half]
        elif self.plane == QuadPlane.ZPlane:
            return [self.tr_position[0] + (-1)**(i//2) * scale_half, self.tr_position[1] + (-1)**(i%2) * scale_half, self.tr_position[2]]

    def initialize(self, parent, index):
        self.scale = parent.scale / 2
        self.level = parent.level + 1
        self.plane = parent.plane
        self.parent = parent
        self.planet = parent.planet
        self.index = index
        self.position = parent.position
        self.planet.quads.append(self)
        self.start_mesh_generation()

import numpy as np
import math

class ModuleType:
    Heightmap = -2
    Noise = -1
    Select = auto()
    Curve = auto()
    Blend = auto()
    Remap = auto()
    Add = auto()
    Subtract = auto()
    Multiply = auto()
    Min = auto()
    Max = auto()
    Scale = auto()
    ScaleBias = auto()
    Abs = auto()
    Invert = auto()
    Clamp = auto()
    Const = auto()
    Terrace = auto()

class Module:
    def __init__(self):
        self.inputs = []
        self.parameters = []
        self.op_type = None

    def get_noise(self, x, y, z):
        raise NotImplementedError

    def __eq__(self, other):
        if not isinstance(other, Module):
            return False
        if self.op_type != other.op_type:
            return False
        if not np.allclose(self.parameters, other.parameters):
            return False
        if len(self.inputs) != len(other.inputs):
            return False
        return all(i1 == i2 for i1, i2 in zip(self.inputs, other.inputs))

    def __hash__(self):
        return hash((self.op_type, tuple(self.parameters), tuple(self.inputs)))

class Select(Module):
    def __init__(self, terrain_type, noise1, noise2, fall_off=0.175, min_val=-1.0, max_val=0.0):
        super().__init__()
        self.op_type = ModuleType.Select
        self.inputs = [noise1, noise2, terrain_type]
        self.parameters = [fall_off, min_val, max_val]

    def get_noise(self, x, y, z):
        cv = self.inputs[2].get_noise(x, y, z)
        if self.parameters[0] > 0:
            if cv < (self.parameters[1] - self.parameters[0]):
                return self.inputs[0].get_noise(x, y, z)
            if cv < (self.parameters[1] + self.parameters[0]):
                lc = self.parameters[1] - self.parameters[0]
                uc = self.parameters[1] + self.parameters[0]
                a = self.map_cubic_s_curve((cv - lc) / (uc - lc))
                return np.interp(a, [self.inputs[0].get_noise(x, y, z), self.inputs[1].get_noise(x, y, z)])
            if cv < (self.parameters[2] - self.parameters[0]):
                return self.inputs[1].get_noise(x, y, z)
            if cv < (self.parameters[2] + self.parameters[0]):
                lc = self.parameters[2] - self.parameters[0]
                uc = self.parameters[2] + self.parameters[0]
                a = self.map_cubic_s_curve((cv - lc) / (uc - lc))
                return np.interp(a, [self.inputs[1].get_noise(x, y, z), self.inputs[0].get_noise(x, y, z)])
            return self.inputs[0].get_noise(x, y, z)
        if cv < self.parameters[1] or cv > self.parameters[2]:
            return self.inputs[0].get_noise(x, y, z)
        return self.inputs[1].get_noise(x, y, z)

    @staticmethod
    def map_cubic_s_curve(value):
        return value * value * (3 - 2 * value)

class Const(Module):
    def __init__(self, constant):
        super().__init__()
        self.op_type = ModuleType.Const
        self.parameters = [constant]

    def get_noise(self, x, y, z):
        return self.parameters[0]

class Add(Module):
    def __init__(self, module1, module2):
        super().__init__()
        self.op_type = ModuleType.Add
        self.inputs = [module1, module2]

    def get_noise(self, x, y, z):
        return self.inputs[0].get_noise(x, y, z) + self.inputs[1].get_noise(x, y, z)

class Multiply(Module):
    def __init__(self, module1, module2):
        super().__init__()
        self.op_type = ModuleType.Multiply
        self.inputs = [module1, module2]

    def get_noise(self, x, y, z):
        return self.inputs[0].get_noise(x, y, z) * self.inputs[1].get_noise(x, y, z)

class Scale(Module):
    def __init__(self, module1, scale):
        super().__init__()
        self.op_type = ModuleType.Scale
        self.inputs = [module1]
        self.parameters = [scale]

    def get_noise(self, x, y, z):
        return self.inputs[0].get_noise(x, y, z) * self.parameters[0]

class ScaleBias(Module):
    def __init__(self, module1, scale, bias):
        super().__init__()
        self.op_type = ModuleType.ScaleBias
        self.inputs = [module1]
        self.parameters = [scale, bias]

    def get_noise(self, x, y, z):
        return self.inputs[0].get_noise(x, y, z) * self.parameters[0] + self.parameters[1]

class Abs(Module):
    def __init__(self, module1):
        super().__init__()
        self.op_type = ModuleType.Abs
        self.inputs = [module1]

    def get_noise(self, x, y, z):
        return abs(self.inputs[0].get_noise(x, y, z))

class Clamp(Module):
    def __init__(self, module1, min_val, max_val):
        super().__init__()
        self.op_type = ModuleType.Clamp
        self.inputs = [module1]
        self.parameters = [min_val, max_val]

    def get_noise(self, x, y, z):
        return np.clip(self.inputs[0].get_noise(x, y, z), self.parameters[0], self.parameters[1])

class Curve(Module):
    def __init__(self, module1, curve):
        super().__init__()
        self.op_type = ModuleType.Curve
        self.inputs = [module1]
        self.curve = curve

    def get_noise(self, x, y, z):
        return np.interp(self.inputs[0].get_noise(x, y, z), self.curve[0], self.curve[1])

class Subtract(Module):
    def __init__(self, module1, module2):
        super().__init__()
        self.op_type = ModuleType.Subtract
        self.inputs = [module1, module2]

    def get_noise(self, x, y, z):
        return self.inputs[0].get_noise(x, y, z) - self.inputs[1].get_noise(x, y, z)

class Blend(Module):
    def __init__(self, module1, module2, bias=0.5):
        super().__init__()
        self.op_type = ModuleType.Blend
        self.inputs = [module1, module2]
        self.parameters = [bias]

    def get_noise(self, x, y, z):
        a = self.inputs[0].get_noise(x, y, z)
        b = self.inputs[1].get_noise(x, y, z)
        return a + self.parameters[0] * (b - a)

class Remap(Module):
    def __init__(self, module1, scale_x, scale_y, scale_z, offset_x, offset_y, offset_z):
        super().__init__()
        self.op_type = ModuleType.Remap
        self.inputs = [module1]
        self.parameters = [scale_x, scale_y, scale_z, offset_x, offset_y, offset_z]

    def get_noise(self, x, y, z):
        return self.inputs[0].get_noise(
            x * self.parameters[0] + self.parameters[3],
            y * self.parameters[1] + self.parameters[4],
            z * self.parameters[2] + self.parameters[5]
        )

class Min(Module):
    def __init__(self, module1, module2):
        super().__init__()
        self.op_type = ModuleType.Min
        self.inputs = [module1, module2]

    def get_noise(self, x, y, z):
        return min(self.inputs[0].get_noise(x, y, z), self.inputs[1].get_noise(x, y, z))

class Max(Module):
    def __init__(self, module1, module2):
        super().__init__()
        self.op_type = ModuleType.Max
        self.inputs = [module1, module2]

    def get_noise(self, x, y, z):
        return max(self.inputs[0].get_noise(x, y, z), self.inputs[1].get_noise(x, y, z))

class Invert(Module):
    def __init__(self, module1):
        super().__init__()
        self.op_type = ModuleType.Invert
        self.inputs = [module1]

    def get_noise(self, x, y, z):
        return -self.inputs[0].get_noise(x, y, z)

class Terrace(Module):
    def __init__(self, module1, control_points):
        super().__init__()
        self.op_type = ModuleType.Terrace
        if len(control_points) < 2:
            raise ValueError("Two or more control points must be specified.")
        self.parameters = control_points
        self.inputs = [module1]

    def get_noise(self, x, y, z):
        source_value = self.inputs[0].get_noise(x, y, z)
        control_point_count = len(self.parameters)
        index_pos = next((i for i in range(control_point_count) if source_value < self.parameters[i]), control_point_count - 1)
        index0 = max(index_pos - 1, 0)
        index1 = min(index_pos, control_point_count - 1)
        if index0 == index1:
            return self.parameters[index1]
        value0 = self.parameters[index0]
        value1 = self.parameters[index1]
        alpha = (source_value - value0) / (value1 - value0)
        alpha *= alpha
        return np.interp(alpha, [value0, value1])

class HeightmapModule(Module):
    def __init__(self, heightmap_data, use_bicubic_interpolation):
        super().__init__()
        self.op_type = ModuleType.Heightmap
        self.heightmap_data = heightmap_data
        self.use_bicubic_interpolation = use_bicubic_interpolation
        self.heightmap = None

    def get_noise(self, x, y, z):
        if self.heightmap is None:
            return 0
        return self.heightmap.get_pos_interpolated(x, y, z) * 2 - 1

    def init(self):
        self.heightmap = Heightmap(self.heightmap_data, self.use_bicubic_interpolation)

import numpy as np
import math

class Vector3d:
    kEpsilon = 1E-05

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z
        else:
            raise IndexError("Invalid index!")

    def __setitem__(self, index, value):
        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value
        elif index == 2:
            self.z = value
        else:
            raise IndexError("Invalid Vector3d index!")

    @property
    def normalized(self):
        return Vector3d.normalize(self)

    @property
    def magnitude(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    @property
    def sqr_magnitude(self):
        return self.x ** 2 + self.y ** 2 + self.z ** 2

    @staticmethod
    def zero():
        return Vector3d(0.0, 0.0, 0.0)

    @staticmethod
    def one():
        return Vector3d(1.0, 1.0, 1.0)

    @staticmethod
    def forward():
        return Vector3d(0.0, 0.0, 1.0)

    @staticmethod
    def back():
        return Vector3d(0.0, 0.0, -1.0)

    @staticmethod
    def up():
        return Vector3d(0.0, 1.0, 0.0)

    @staticmethod
    def down():
        return Vector3d(0.0, -1.0, 0.0)

    @staticmethod
    def left():
        return Vector3d(-1.0, 0.0, 0.0)

    @staticmethod
    def right():
        return Vector3d(1.0, 0.0, 0.0)

    def __add__(self, other):
        return Vector3d(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3d(self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self):
        return Vector3d(-self.x, -self.y, -self.z)

    def __mul__(self, d):
        return Vector3d(self.x * d, self.y * d, self.z * d)

    def __truediv__(self, d):
        return Vector3d(self.x / d, self.y / d, self.z / d)

    def __eq__(self, other):
        return Vector3d.sqr_magnitude(self - other) < Vector3d.kEpsilon

    def __ne__(self, other):
        return not self == other

    @staticmethod
    def lerp(from_v, to_v, t):
        t = Vector3d.clamp01(t)
        return from_v + (to_v - from_v) * t

    @staticmethod
    def slerp(from_v, to_v, t):
        from_v = np.array([from_v.x, from_v.y, from_v.z], dtype=np.float32)
        to_v = np.array([to_v.x, to_v.y, to_v.z], dtype=np.float32)
        result = np.slerp(from_v, to_v, t)
        return Vector3d(result[0], result[1], result[2])

    @staticmethod
    def ortho_normalize(normal, tangent):
        normal = np.array([normal.x, normal.y, normal.z], dtype=np.float32)
        tangent = np.array([tangent.x, tangent.y, tangent.z], dtype=np.float32)
        normal, tangent = np.linalg.qr(np.column_stack((normal, tangent)))
        return Vector3d(*normal[:, 0]), Vector3d(*tangent[:, 1])

    @staticmethod
    def move_towards(current, target, max_distance_delta):
        to_vector = target - current
        dist = to_vector.magnitude
        if dist <= max_distance_delta or dist == 0.0:
            return target
        return current + to_vector / dist * max_distance_delta

    @staticmethod
    def smooth_damp(current, target, current_velocity, smooth_time, max_speed=np.inf, delta_time=1/60):
        smooth_time = max(0.0001, smooth_time)
        omega = 2.0 / smooth_time
        x = omega * delta_time
        exp = 1.0 / (1.0 + x + 0.48 * x ** 2 + 0.235 * x ** 3)
        change = current - target
        original_to = target

        max_change = max_speed * smooth_time
        change = Vector3d.clamp_magnitude(change, max_change)
        target = current - change

        temp = (current_velocity + omega * change) * delta_time
        current_velocity = (current_velocity - omega * temp) * exp
        output = target + (change + temp) * exp

        if Vector3d.dot(original_to - current, output - original_to) > 0:
            output = original_to
            current_velocity = (output - original_to) / delta_time

        return output, current_velocity

    def set(self, new_x, new_y, new_z):
        self.x = new_x
        self.y = new_y
        self.z = new_z

    @staticmethod
    def scale(a, b):
        return Vector3d(a.x * b.x, a.y * b.y, a.z * b.z)

    def scale(self, scale):
        self.x *= scale.x
        self.y *= scale.y
        self.z *= scale.z

    @staticmethod
    def cross(lhs, rhs):
        return Vector3d(lhs.y * rhs.z - lhs.z * rhs.y, lhs.z * rhs.x - lhs.x * rhs.z, lhs.x * rhs.y - lhs.y * rhs.x)

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __eq__(self, other):
        return isinstance(other, Vector3d) and self.x == other.x and self.y == other.y and self.z == other.z

    @staticmethod
    def reflect(in_direction, in_normal):
        return in_direction - 2 * Vector3d.dot(in_normal, in_direction) * in_normal

    @staticmethod
    def normalize(value):
        mag = value.magnitude
        if mag > Vector3d.kEpsilon:
            return value / mag
        return Vector3d.zero()

    def normalize_self(self):
        mag = self.magnitude
        if mag > Vector3d.kEpsilon:
            self /= mag
        else:
            self.set(0.0, 0.0, 0.0)

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

    @staticmethod
    def dot(lhs, rhs):
        return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z

    @staticmethod
    def project(vector, on_normal):
        num = Vector3d.dot(on_normal, on_normal)
        if num < 1.40129846432482E-45:
            return Vector3d.zero()
        return on_normal * (Vector3d.dot(vector, on_normal) / num)

    @staticmethod
    def exclude(exclude_this, from_that):
        return from_that - Vector3d.project(from_that, exclude_this)

    @staticmethod
    def angle(from_v, to_v):
        return math.acos(Vector3d.clamp(Vector3d.dot(from_v.normalized, to_v.normalized), -1, 1)) * 57.29578

    @staticmethod
    def distance(a, b):
        return (a - b).magnitude

    @staticmethod
    def clamp_magnitude(vector, max_length):
        if vector.sqr_magnitude > max_length * max_length:
            return vector.normalized * max_length
        return vector

    @staticmethod
    def magnitude(a):
        return a.magnitude

    @staticmethod
    def sqr_magnitude(a):
        return a.sqr_magnitude

    @staticmethod
    def min(lhs, rhs):
        return Vector3d(min(lhs.x, rhs.x), min(lhs.y, rhs.y), min(lhs.z, rhs.z))

    @staticmethod
    def max(lhs, rhs):
        return Vector3d(max(lhs.x, rhs.x), max(lhs.y, rhs.y), max(lhs.z, rhs.z))

    @staticmethod
    def clamp(value, min_val, max_val):
        return max(min_val, min(value, max_val))

    @staticmethod
    def clamp01(value):
        return Vector3d.clamp(value, 0, 1)

import math

class Mathd:
    @staticmethod
    def Sin(d):
        return math.sin(d)

    @staticmethod
    def Cos(d):
        return math.cos(d)

    @staticmethod
    def Tan(d):
        return math.tan(d)

    @staticmethod
    def Asin(d):
        return math.asin(d)

    @staticmethod
    def Acos(d):
        return math.acos(d)

    @staticmethod
    def Atan(d):
        return math.atan(d)

    @staticmethod
    def Atan2(y, x):
        return math.atan2(y, x)

    @staticmethod
    def Sqrt(d):
        return math.sqrt(d)

    @staticmethod
    def Abs(d):
        return abs(d)

    @staticmethod
    def Min(a, b):
        return min(a, b)

    @staticmethod
    def MinList(values):
        if len(values) == 0:
            return 0.0
        return min(values)

    @staticmethod
    def Max(a, b):
        return max(a, b)

    @staticmethod
    def MaxList(values):
        if len(values) == 0:
            return 0.0
        return max(values)

    @staticmethod
    def Pow(d, p):
        return math.pow(d, p)

    @staticmethod
    def Exp(power):
        return math.exp(power)

    @staticmethod
    def Log(d, p=math.e):
        return math.log(d, p)

    @staticmethod
    def Log10(d):
        return math.log10(d)

    @staticmethod
    def Ceil(d):
        return math.ceil(d)

    @staticmethod
    def Floor(d):
        return math.floor(d)

    @staticmethod
    def Round(d):
        return round(d)

    @staticmethod
    def CeilToInt(d):
        return math.ceil(d)

    @staticmethod
    def FloorToInt(d):
        return math.floor(d)

    @staticmethod
    def RoundToInt(d):
        return round(d)

    @staticmethod
    def Sign(d):
        return 1.0 if d >= 0.0 else -1.0

    @staticmethod
    def Clamp(value, min_val, max_val):
        return max(min_val, min(value, max_val))

    @staticmethod
    def Clamp01(value):
        return Mathd.Clamp(value, 0.0, 1.0)

    @staticmethod
    def Lerp(from_val, to_val, t):
        return from_val + (to_val - from_val) * Mathd.Clamp01(t)

    @staticmethod
    def LerpAngle(a, b, t):
        num = Mathd.Repeat(b - a, 360.0)
        if num > 180.0:
            num -= 360.0
        return a + num * Mathd.Clamp01(t)

    @staticmethod
    def MoveTowards(current, target, maxDelta):
        if Mathd.Abs(target - current) <= maxDelta:
            return target
        return current + Mathd.Sign(target - current) * maxDelta

    @staticmethod
    def MoveTowardsAngle(current, target, maxDelta):
        target = current + Mathd.DeltaAngle(current, target)
        return Mathd.MoveTowards(current, target, maxDelta)

    @staticmethod
    def SmoothStep(from_val, to_val, t):
        t = Mathd.Clamp01(t)
        t = (-2.0 * t * t * t + 3.0 * t * t)
        return to_val * t + from_val * (1.0 - t)

    @staticmethod
    def Gamma(value, absmax, gamma):
        flag = value < 0.0
        num1 = Mathd.Abs(value)
        if num1 > absmax:
            return -num1 if flag else num1
        num2 = Mathd.Pow(num1 / absmax, gamma) * absmax
        return -num2 if flag else num2

    @staticmethod
    def Approximately(a, b):
        return Mathd.Abs(b - a) < Mathd.Max(1E-06 * Mathd.Max(Mathd.Abs(a), Mathd.Abs(b)), 1.121039E-44)

    @staticmethod
    def SmoothDamp(current, target, current_velocity, smooth_time, max_speed=float('inf'), delta_time=1/60):
        smooth_time = max(0.0001, smooth_time)
        omega = 2.0 / smooth_time
        x = omega * delta_time
        exp = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
        change = current - target
        original_to = target

        max_change = max_speed * smooth_time
        change = Mathd.Clamp(change, -max_change, max_change)
        target = current - change

        temp = (current_velocity + omega * change) * delta_time
        current_velocity = (current_velocity - omega * temp) * exp
        output = target + (change + temp) * exp

        if (original_to - current > 0.0) == (output > original_to):
            output = original_to
            current_velocity = (output - original_to) / delta_time

        return output, current_velocity

    @staticmethod
    def SmoothDampAngle(current, target, current_velocity, smooth_time, max_speed=float('inf'), delta_time=1/60):
        target = current + Mathd.DeltaAngle(current, target)
        return Mathd.SmoothDamp(current, target, current_velocity, smooth_time, max_speed, delta_time)

    @staticmethod
    def Repeat(t, length):
        return t - Mathd.Floor(t / length) * length

    @staticmethod
    def PingPong(t, length):
        t = Mathd.Repeat(t, length * 2.0)
        return length - Mathd.Abs(t - length)

    @staticmethod
    def InverseLerp(from_val, to_val, value):
        if from_val < to_val:
            if value < from_val:
                return 0.0
            if value > to_val:
                return 1.0
            value -= from_val
            value /= to_val - from_val
            return value
        if from_val <= to_val:
            return 0.0
        if value < to_val:
            return 1.0
        if value > from_val:
            return 0.0
        return 1.0 - (value - to_val) / (from_val - to_val)

    @staticmethod
    def DeltaAngle(current, target):
        num = Mathd.Repeat(target - current, 360.0)
        if num > 180.0:
            num -= 360.0
        return num

import numpy as np
from PIL import Image

# Assuming a hypothetical pygpu module for GPU operations
import pygpu

class CONSTANTS:
    NUM_THREADS = 8
    TRANSMITTANCE_WIDTH = 256
    TRANSMITTANCE_HEIGHT = 64
    IRRADIANCE_WIDTH = 64
    IRRADIANCE_HEIGHT = 16
    SCATTERING_WIDTH = 128
    SCATTERING_HEIGHT = 32
    SCATTERING_DEPTH = 32


class TextureBuffer:
    def __init__(self, half_precision):
        # 16F precision for the transmittance gives artifacts. Always use full.
        # Also using full for irradiance as the original code did.

        self.TransmittanceArray = self.new_texture_2d_array(
            CONSTANTS.TRANSMITTANCE_WIDTH,
            CONSTANTS.TRANSMITTANCE_HEIGHT,
            False
        )

        self.IrradianceArray = self.new_texture_2d_array(
            CONSTANTS.IRRADIANCE_WIDTH,
            CONSTANTS.IRRADIANCE_HEIGHT,
            False
        )

        self.ScatteringArray = self.new_texture_3d_array(
            CONSTANTS.SCATTERING_WIDTH,
            CONSTANTS.SCATTERING_HEIGHT,
            CONSTANTS.SCATTERING_DEPTH,
            half_precision
        )

        self.OptionalSingleMieScatteringArray = self.new_texture_3d_array(
            CONSTANTS.SCATTERING_WIDTH,
            CONSTANTS.SCATTERING_HEIGHT,
            CONSTANTS.SCATTERING_DEPTH,
            half_precision
        )

        self.DeltaIrradianceTexture = self.new_render_texture_2d(
            CONSTANTS.IRRADIANCE_WIDTH,
            CONSTANTS.IRRADIANCE_HEIGHT,
            False
        )

        self.DeltaRayleighScatteringTexture = self.new_render_texture_3d(
            CONSTANTS.SCATTERING_WIDTH,
            CONSTANTS.SCATTERING_HEIGHT,
            CONSTANTS.SCATTERING_DEPTH,
            half_precision
        )

        self.DeltaMieScatteringTexture = self.new_render_texture_3d(
            CONSTANTS.SCATTERING_WIDTH,
            CONSTANTS.SCATTERING_HEIGHT,
            CONSTANTS.SCATTERING_DEPTH,
            half_precision
        )

        self.DeltaScatteringDensityTexture = self.new_render_texture_3d(
            CONSTANTS.SCATTERING_WIDTH,
            CONSTANTS.SCATTERING_HEIGHT,
            CONSTANTS.SCATTERING_DEPTH,
            half_precision
        )

        # delta_multiple_scattering_texture is only needed to compute scattering
        # order 3 or more, while delta_rayleigh_scattering_texture and
        # delta_mie_scattering_texture are only needed to compute double scattering.
        # Therefore, to save memory, we can store delta_rayleigh_scattering_texture
        # and delta_multiple_scattering_texture in the same GPU texture.
        self.DeltaMultipleScatteringTexture = self.DeltaRayleighScatteringTexture

    def release(self):
        self.release_texture(self.DeltaIrradianceTexture)
        self.release_texture(self.DeltaRayleighScatteringTexture)
        self.release_texture(self.DeltaMieScatteringTexture)
        self.release_texture(self.DeltaScatteringDensityTexture)
        self.release_array(self.TransmittanceArray)
        self.release_array(self.IrradianceArray)
        self.release_array(self.ScatteringArray)
        self.release_array(self.OptionalSingleMieScatteringArray)

    def clear(self, compute):
        self.clear_texture(compute, self.DeltaIrradianceTexture)
        self.clear_texture(compute, self.DeltaRayleighScatteringTexture)
        self.clear_texture(compute, self.DeltaMieScatteringTexture)
        self.clear_texture(compute, self.DeltaScatteringDensityTexture)
        self.clear_array(compute, self.TransmittanceArray)
        self.clear_array(compute, self.IrradianceArray)
        self.clear_array(compute, self.ScatteringArray)
        self.clear_array(compute, self.OptionalSingleMieScatteringArray)

    def release_texture(self, tex):
        if tex is None:
            return
        tex.release()

    def release_array(self, arr):
        if arr is None:
            return

        for tex in arr:
            if tex is not None:
                tex.release()

    def new_texture_2d_array(self, width, height, half_precision):
        arr = [
            self.new_render_texture_2d(width, height, half_precision),
            self.new_render_texture_2d(width, height, half_precision)
        ]
        return arr

    def new_texture_3d_array(self, width, height, depth, half_precision):
        arr = [
            self.new_render_texture_3d(width, height, depth, half_precision),
            self.new_render_texture_3d(width, height, depth, half_precision)
        ]
        return arr

    def clear_array(self, compute, arr):
        if arr is None:
            return

        for tex in arr:
            self.clear_texture(compute, tex)

    def clear_texture(self, compute, tex):
        if tex is None:
            return

        NUM_THREADS = CONSTANTS.NUM_THREADS

        if tex.dimension == "3D":
            width = tex.width
            height = tex.height
            depth = tex.depth

            kernel = compute.find_kernel("ClearTex3D")
            compute.set_texture(kernel, "targetWrite3D", tex)
            compute.dispatch(kernel, width // NUM_THREADS, height // NUM_THREADS, depth // NUM_THREADS)
        else:
            width = tex.width
            height = tex.height

            kernel = compute.find_kernel("ClearTex2D")
            compute.set_texture(kernel, "targetWrite2D", tex)
            compute.dispatch(kernel, width // NUM_THREADS, height // NUM_THREADS, 1)

    @staticmethod
    def new_render_texture_2d(width, height, half_precision):
        format = "ARGBFloat"

        # Half not always supported.
        if half_precision and pygpu.supports_render_texture_format("ARGBHalf"):
            format = "ARGBHalf"

        map = pygpu.RenderTexture(width, height, 0, format, "Linear")
        map.filter_mode = "Bilinear"
        map.wrap_mode = "Clamp"
        map.use_mip_map = False
        map.enable_random_write = True
        map.create()

        return map

    @staticmethod
    def new_render_texture_3d(width, height, depth, half_precision):
        format = "ARGBFloat"

        # Half not always supported.
        if half_precision and pygpu.supports_render_texture_format("ARGBHalf"):
            format = "ARGBHalf"

        map = pygpu.RenderTexture(width, height, 0, format, "Linear")
        map.volume_depth = depth
        map.dimension = "3D"
        map.filter_mode = "Bilinear"
        map.wrap_mode = "Clamp"
        map.use_mip_map = False
        map.enable_random_write = True
        map.create()

        return map

    @staticmethod
    def new_texture_2d(width, height, half_precision):
        format = "RGBAFloat"

        # Half not always supported.
        if half_precision and pygpu.supports_texture_format("RGBAHalf"):
            format = "RGBAHalf"

        map = pygpu.Texture2D(width, height, format, False, True)
        map.filter_mode = "Bilinear"
        map.wrap_mode = "Clamp"

        return map

    @staticmethod
    def new_texture_3d(width, height, depth, half_precision):
        format = "RGBAFloat"

        # Half not always supported.
        if half_precision and pygpu.supports_texture_format("RGBAHalf"):
            format = "RGBAHalf"

        map = pygpu.Texture3D(width, height, depth, format, False)
        map.filter_mode = "Bilinear"
        map.wrap_mode = "Clamp"

        return map

import numpy as np
from PIL import Image, ImageDraw
from abc import ABC, abstractmethod

class ITextureProvider(ABC):
    @abstractmethod
    def evaluate_texture(self, height, normalized):
        pass

class TextureProviderNone(ITextureProvider):
    def evaluate_texture(self, height, normalized):
        return [0.0] * 6

class TextureProviderGradient(ITextureProvider):
    MAX_NUM_INDICES = 6

    def __init__(self, heights=None, ids=None):
        self.heights = heights if heights else [0.0, 0.01, 0.02, 0.75, 1.0]
        self.ids = ids if ids else [0, 1, 2, 3, 4, 5]

    def evaluate_texture(self, height, normalized):
        height = np.clip(height, 0, 1)
        index = np.searchsorted(self.heights, height)

        index1 = np.clip(index - 1, 0, len(self.heights) - 1)
        index2 = np.clip(index, 0, len(self.heights) - 1)

        result = [0.0] * self.MAX_NUM_INDICES
        if self.ids[index1] == self.ids[index2]:
            result[self.ids[index1]] = 1.0
            return result

        height = (height - self.heights[index1]) / (self.heights[index2] - self.heights[index1])
        result[self.ids[index1]] = 1.0 - height
        result[self.ids[index2]] = height

        return result

    def evaluate_color(self, height, sequence):
        value = self.evaluate_texture(height, np.zeros(3))
        return float_array_to_color(value, sequence)

    def get_sample_texture(self, colors, width=256, height=32):
        pixels = np.zeros((height, width, 4), dtype=np.uint8)
        zero = np.zeros(3)

        for x in range(width):
            value = self.evaluate_texture(x / (width - 1), zero)
            col = float_array_to_color(value, colors)

            for y in range(height):
                pixels[y, x] = col

        texture = Image.fromarray(pixels, 'RGBA')
        return texture

def float_array_to_color(value, colors):
    result_color = np.zeros(4, dtype=np.uint8)
    for i, val in enumerate(value):
        result_color += np.array(colors[i]) * val
    return np.clip(result_color, 0, 255).astype(np.uint8)

class TextureProviderRange(ITextureProvider):
    def __init__(self):
        self.ranges = [np.array([0.0, 0.666667]), np.array([0.333333, 1.0])]
        self.textures = [0, 1]

    def evaluate_texture(self, height, normalized):
        result = [0.0] * 6
        indices = [i for i, range in enumerate(self.ranges) if range[0] <= height <= range[1]]

        if not indices:
            return result

        if len(indices) == 1 or self.textures[indices[0]] == self.textures[indices[1]]:
            result[self.textures[indices[0]]] = 1.0
            return result

        sum_of_distances = 0
        for i in indices:
            dist = min(abs(self.ranges[i][0] - height), abs(self.ranges[i][1] - height)) / abs(self.ranges[i][0] - self.ranges[i][1])
            result[self.textures[i]] = dist
            sum_of_distances += dist

        result = [r / sum_of_distances for r in result]
        return result

    def get_sample_texture(self, colors, width=256, height=32):
        pixels = np.zeros((height, width, 4), dtype=np.uint8)
        zero = np.zeros(3)

        for x in range(width):
            value = self.evaluate_texture(x / (width - 1), zero)
            col = float_array_to_color(value, colors)

            for y in range(height):
                pixels[y, x] = col

        texture = Image.fromarray(pixels, 'RGBA')
        return texture

# Assuming the presence of a Heightmap class and utility functions like MathFunctions in the original code.
import numpy as np
from PIL import Image
import math

class Planet:
    def __init__(self, radius, height_scale):
        self.radius = radius
        self.height_scale = height_scale

class DensityProfileLayer:
    def __init__(self, name, width, density, exp_term, linear_term, constant_term):
        self.name = name
        self.width = width
        self.density = density
        self.exp_term = exp_term
        self.linear_term = linear_term
        self.constant_term = constant_term

class Model:
    def __init__(self):
        self.half_precision = False
        self.combine_scattering_textures = False
        self.use_luminance = None
        self.wavelengths = []
        self.solar_irradiance = []
        self.sun_angular_radius = 0.0
        self.bottom_radius = 0.0
        self.top_radius = 0.0
        self.rayleigh_density = None
        self.rayleigh_scattering = []
        self.mie_density = None
        self.mie_scattering = []
        self.mie_extinction = []
        self.mie_phase_function_g = 0.0
        self.absorption_density = []
        self.absorption_extinction = []
        self.ground_albedo = []
        self.max_sun_zenith_angle = 0.0
        self.length_unit_in_meters = 0.0

    def init(self, compute, num_scattering_orders):
        pass  # Placeholder for shader operations

    def bind_to_material(self, material):
        pass  # Placeholder for binding model to material

    def release(self):
        pass  # Placeholder for resource release

    def convert_spectrum_to_linear_srgb(self):
        white_point_r, white_point_g, white_point_b = 1.0, 1.0, 1.0
        return white_point_r, white_point_g, white_point_b

class AtmospherePostProcessing:
    kSunAngularRadius = 0.00935 / 2.0

    def __init__(self, planet, sun, compute_shader, material):
        self.planet = planet
        self.sun = sun
        self.top_bottom_radius_ratio = 1.01711
        self.sea_level_height = 0
        self.use_constant_solar_spectrum = False
        self.use_ozone = True
        self.use_combined_textures = True
        self.use_half_precision = False
        self.do_white_balance = False
        self.use_luminance = None
        self.exposure = 10.0
        self.compute_shader = compute_shader
        self.material = material

        self.m_model = None
        self.m_camera = None
        self.kBottomRadius = 6371000.0
        self.kLengthUnitInMeters = 0.0

    def awake(self):
        inv_height_scale = 1 / self.planet.height_scale
        self.kLengthUnitInMeters = self.kBottomRadius / (self.planet.radius * ((inv_height_scale + self.sea_level_height) / inv_height_scale))

        kLambdaMin = 360
        kLambdaMax = 830

        kSolarIrradiance = [
            1.11776, 1.14259, 1.01249, 1.14716, 1.72765, 1.73054, 1.6887, 1.61253,
            1.91198, 2.03474, 2.02042, 2.02212, 1.93377, 1.95809, 1.91686, 1.8298,
            1.8685, 1.8931, 1.85149, 1.8504, 1.8341, 1.8345, 1.8147, 1.78158, 1.7533,
            1.6965, 1.68194, 1.64654, 1.6048, 1.52143, 1.55622, 1.5113, 1.474, 1.4482,
            1.41018, 1.36775, 1.34188, 1.31429, 1.28303, 1.26758, 1.2367, 1.2082,
            1.18737, 1.14683, 1.12362, 1.1058, 1.07124, 1.04992
        ]

        kOzoneCrossSection = [
            1.18e-27, 2.182e-28, 2.818e-28, 6.636e-28, 1.527e-27, 2.763e-27, 5.52e-27,
            8.451e-27, 1.582e-26, 2.316e-26, 3.669e-26, 4.924e-26, 7.752e-26, 9.016e-26,
            1.48e-25, 1.602e-25, 2.139e-25, 2.755e-25, 3.091e-25, 3.5e-25, 4.266e-25,
            4.672e-25, 4.398e-25, 4.701e-25, 5.019e-25, 4.305e-25, 3.74e-25, 3.215e-25,
            2.662e-25, 2.238e-25, 1.852e-25, 1.473e-25, 1.209e-25, 9.423e-26, 7.455e-26,
            6.566e-26, 5.105e-26, 4.15e-26, 4.228e-26, 3.237e-26, 2.451e-26, 2.801e-26,
            2.534e-26, 1.624e-26, 1.465e-26, 2.078e-26, 1.383e-26, 7.105e-27
        ]

        kDobsonUnit = 2.687e20
        kMaxOzoneNumberDensity = 300.0 * kDobsonUnit / 15000.0
        kConstantSolarIrradiance = 1.5
        kTopRadius = self.kBottomRadius * self.top_bottom_radius_ratio
        kRayleigh = 1.24062e-6
        kRayleighScaleHeight = 8000.0
        kMieScaleHeight = 1200.0
        kMieAngstromAlpha = 0.0
        kMieAngstromBeta = 5.328e-3
        kMieSingleScatteringAlbedo = 0.9
        kMiePhaseFunctionG = 0.8
        kGroundAlbedo = 0.1
        max_sun_zenith_angle = (102.0 if self.use_half_precision else 120.0) / 180.0 * np.pi

        rayleigh_layer = DensityProfileLayer("rayleigh", 0.0, 1.0, -1.0 / kRayleighScaleHeight, 0.0, 0.0)
        mie_layer = DensityProfileLayer("mie", 0.0, 1.0, -1.0 / kMieScaleHeight, 0.0, 0.0)

        ozone_density = [
            DensityProfileLayer("absorption0", 25000.0, 0.0, 0.0, 1.0 / 15000.0, -2.0 / 3.0),
            DensityProfileLayer("absorption1", 0.0, 0.0, 0.0, -1.0 / 15000.0, 8.0 / 3.0)
        ]

        wavelengths = []
        solar_irradiance = []
        rayleigh_scattering = []
        mie_scattering = []
        mie_extinction = []
        absorption_extinction = []
        ground_albedo = []

        for l in range(kLambdaMin, kLambdaMax + 1, 10):
            lambda_ = l * 1e-3  # micro-meters
            mie = kMieAngstromBeta / kMieScaleHeight * math.pow(lambda_, -kMieAngstromAlpha)

            wavelengths.append(l)
            solar_irradiance.append(kConstantSolarIrradiance if self.use_constant_solar_spectrum else kSolarIrradiance[(l - kLambdaMin) // 10])
            rayleigh_scattering.append(kRayleigh * math.pow(lambda_, -4))
            mie_scattering.append(mie * kMieSingleScatteringAlbedo)
            mie_extinction.append(mie)
            absorption_extinction.append(kMaxOzoneNumberDensity * kOzoneCrossSection[(l - kLambdaMin) // 10] if self.use_ozone else 0.0)
            ground_albedo.append(kGroundAlbedo)

        self.m_model = Model()
        self.m_model.half_precision = self.use_half_precision
        self.m_model.combine_scattering_textures = self.use_combined_textures
        self.m_model.use_luminance = self.use_luminance
        self.m_model.wavelengths = wavelengths
        self.m_model.solar_irradiance = solar_irradiance
        self.m_model.sun_angular_radius = self.kSunAngularRadius
        self.m_model.bottom_radius = self.kBottomRadius
        self.m_model.top_radius = kTopRadius
        self.m_model.rayleigh_density = rayleigh_layer
        self.m_model.rayleigh_scattering = rayleigh_scattering
        self.m_model.mie_density = mie_layer
        self.m_model.mie_scattering = mie_scattering
        self.m_model.mie_extinction = mie_extinction
        self.m_model.mie_phase_function_g = kMiePhaseFunctionG
        self.m_model.absorption_density = ozone_density
        self.m_model.absorption_extinction = absorption_extinction
        self.m_model.ground_albedo = ground_albedo
        self.m_model.max_sun_zenith_angle = max_sun_zenith_angle
        self.m_model.length_unit_in_meters = self.kLengthUnitInMeters

        num_scattering_orders = 4
        self.m_model.init(self.compute_shader, num_scattering_orders)
        self.m_model.bind_to_material(self.material)

        self.set_up()

    def on_destroy(self):
        if self.m_model:
            self.m_model.release()

    def set_up(self):
        inv_height_scale = 1 / self.planet.height_scale
        self.material.set_float("max_terrain_radius", self.planet.radius * (inv_height_scale + 1) / inv_height_scale)
        self.material.set_float("exposure", self.exposure * 1e-5 if self.use_luminance != LUMINANCE.NONE else self.exposure)
        self.material.set_vector("sun_size", (math.tan(self.kSunAngularRadius), math.cos(self.kSunAngularRadius)))

        white_point_r, white_point_g, white_point_b = 1.0, 1.0, 1.0
        if self.do_white_balance:
            white_point_r, white_point_g, white_point_b = self.m_model.convert_spectrum_to_linear_srgb()
            white_point = (white_point_r + white_point_g + white_point_b) / 3.0
            white_point_r /= white_point
            white_point_g /= white_point
            white_point_b /= white_point
        self.material.set_vector("white_point", (white_point_r, white_point_g, white_point_b))

    def on_render_image(self, src, dest):
        p = self.camera.projection_matrix
        p[2, 3] = p[3, 2] = 0.0
        p[3, 3] = 1.0
        clip_to_world = np.linalg.inv(p @ self.camera.world_to_camera_matrix) @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -p[2, 2]], [0, 0, 0, 1]])

        self.material.set_matrix("clip_to_world", clip_to_world)
        self.material.set_vector("earth_center", self.planet.transform.position)
        self.material.set_vector("sun_direction", (self.sun.transform.forward if self.sun else np.array([0, 1, 0])) * -1.0)

        frustum_corners = np.identity(4)
        to_right = self.camera.transform.right * self.c1
        to_top = self.camera.transform.up * self.c2

        top_left = self.camera.transform.forward * self.camera.near_clip_plane - to_right + to_top
        camera_scale = np.linalg.norm(top_left) * self.camera.far_clip_plane / self.camera.near_clip_plane
        top_left = top_left / np.linalg.norm(top_left) * camera_scale

        top_right = self.camera.transform.forward * self.camera.near_clip_plane + to_right + to_top
        top_right = top_right / np.linalg.norm(top_right) * camera_scale

        bottom_right = self.camera.transform.forward * self.camera.near_clip_plane + to_right - to_top
        bottom_right = bottom_right / np.linalg.norm(bottom_right) * camera_scale

        bottom_left = self.camera.transform.forward * self.camera.near_clip_plane - to_right - to_top
        bottom_left = bottom_left / np.linalg.norm(bottom_left) * camera_scale

        frustum_corners[0] = top_left
        frustum_corners[1] = top_right
        frustum_corners[2] = bottom_right
        frustum_corners[3] = bottom_left

        self.material.set_matrix("frustum_corners", frustum_corners)

        self.custom_graphics_blit(src, dest, self.material, 0)

    def custom_graphics_blit(self, source, dest, mat, pass_nr):
        # This is a placeholder for the custom blit operation in Unity's graphics pipeline.
        pass

# Usage
planet = Planet(radius=6371.0, height_scale=1.0)
sun = None  # Placeholder for the sun object
compute_shader = None  # Placeholder for the compute shader
material = None  # Placeholder for the material

atmosphere_post_processing = AtmospherePostProcessing(planet, sun, compute_shader, material)
atmosphere_post_processing.awake()

import numpy as np
from PIL import Image
import random
import math
import os
import pickle

class Utils:
    @staticmethod
    def evaluate_texture(time, texture_heights, texture_ids):
        time = np.clip(time, 0, 1)
        index = 0
        for index in range(len(texture_heights)):
            if time < texture_heights[index]:
                break
        index1 = np.clip(index - 1, 0, len(texture_heights) - 1)
        index2 = np.clip(index, 0, len(texture_heights) - 1)
        
        result = [0.0] * 6
        if texture_ids[index1] == texture_ids[index2]:
            result[texture_ids[index1]] = 1.0
            return result
        
        time = (time - texture_heights[index1]) / (texture_heights[index2] - texture_heights[index1])
        result[texture_ids[index1]] = 1.0 - time
        result[texture_ids[index2]] = time
        return result

    @staticmethod
    def float_array_to_color(floats, colors):
        result = np.array([0, 0, 0, 0], dtype=np.float32)
        for i in range(len(colors)):
            result += floats[i] * np.array(colors[i], dtype=np.float32)
        return tuple(result.astype(np.uint8))

    @staticmethod
    def test_planes_aabb(planes, bounds_min, bounds_max, test_intersection=True, extra_range=0.0):
        if planes is None:
            return False

        vmin, vmax = np.zeros(3), np.zeros(3)
        test_result = 2

        for plane in planes:
            normal, plane_distance = plane[:3], plane[3]
            for axis in range(3):
                if normal[axis] < 0:
                    vmin[axis] = bounds_min[axis]
                    vmax[axis] = bounds_max[axis]
                else:
                    vmin[axis] = bounds_max[axis]
                    vmax[axis] = bounds_min[axis]

            dot1 = np.dot(normal, vmin)
            if dot1 + plane_distance < -extra_range:
                return False

            if test_intersection:
                dot2 = np.dot(normal, vmax)
                if dot2 + plane_distance <= extra_range:
                    test_result = 1

        return test_result > 0

    @staticmethod
    def generate_preview(module, res_x=256, res_y=256):
        preview = Image.new('L', (res_x, res_y))
        for x in range(res_x):
            for y in range(res_y):
                v = (module.get_noise(x / res_x, y / res_y, 0) + 1) / 2
                preview.putpixel((x, y), int(v * 255))
        return preview

    @staticmethod
    def generate_preview_heightmap(module, res_x=256, res_y=256):
        preview = np.zeros((res_x, res_y), dtype=np.float32)
        scale_x = 1 / (2 * res_x)
        scale_y = 1 / (2 * res_y)
        for x in range(res_x):
            for y in range(res_y):
                v = (module.get_noise(x * scale_x - 0.25, y * scale_y - 0.25, 0.75) + 1) / 2
                v = np.clip(v, 0, 1)
                preview[x, y] = v
        return preview

    @staticmethod
    def deserialize_module(file_stream):
        return pickle.load(file_stream)

    @staticmethod
    def deserialize_text_asset(text_asset):
        stream = BytesIO(text_asset.bytes)
        module = pickle.load(stream)
        stream.close()
        Utils.initialize_module_tree(module)
        return module

    @staticmethod
    def deserialize_file(file_name):
        with open(file_name, 'rb') as fs:
            module = pickle.load(fs)
        Utils.initialize_module_tree(module)
        return module

    @staticmethod
    def initialize_module_tree(module):
        if isinstance(module, HeightmapModule):
            module.init()
            if not os.getenv('IS_EDITOR') and os.getenv('IS_PLAYING'):
                module.text_asset_bytes = None

        if module.inputs:
            for input_module in module.inputs:
                Utils.initialize_module_tree(input_module)

    @staticmethod
    def randomize_noise(module):
        if module.op_type == ModuleType.Noise:
            module.set_seed(random.randint(int(np.iinfo(np.int32).min), int(np.iinfo(np.int32).max)))

        if module.inputs:
            for input_module in module.inputs:
                Utils.randomize_noise(input_module)

    @staticmethod
    def save_as_binary(array, filename='filename.bin'):
        floats = np.array(array).flatten()
        with open(filename, 'wb') as file:
            pickle.dump(floats, file)

    @staticmethod
    def save_as_text(array, filename='filename.txt'):
        with open(filename, 'w') as file:
            file.write('{')
            for i, vec in enumerate(array):
                file.write(f'new Vector3({vec[0]}f, {vec[1]}f, {vec[2]}f), ')
                if (i + 1) % 33 == 0:
                    file.write('\n')
            file.write('}')

    @staticmethod
    def array_to_string(array):
        return '{' + ', '.join(map(str, array)) + '}'

    @staticmethod
    def save_as_text_int(array, filename='filename.txt'):
        with open(filename, 'w') as file:
            file.write('{')
            for i, val in enumerate(array):
                file.write(f'{val}, ')
                if (i + 1) % 66 == 0:
                    file.write('\n')
            file.write('}')

class Int2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        else:
            raise IndexError("Int2 index out of range")

    def __setitem__(self, index, value):
        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value
        else:
            raise IndexError("Int2 index out of range")

import math
import numpy as np
from numpy import sin, cos, tan, pi
from scipy.interpolate import interp1d

class AtmosphereMesh:
    kSunAngularRadius = 0.00935 / 2.0
    kLengthUnitInMeters = 6371000.0

    def __init__(self, planet, sun, top_bottom_radius_ratio=1.01711, use_constant_solar_spectrum=False, use_ozone=True, 
                 use_combined_textures=True, use_half_precision=False, do_white_balance=False, use_luminance="NONE", 
                 exposure=10.0, planet_sea_level=0.0, compute_shader=None, material=None):
        self.planet = planet
        self.sun = sun
        self.top_bottom_radius_ratio = top_bottom_radius_ratio
        self.use_constant_solar_spectrum = use_constant_solar_spectrum
        self.use_ozone = use_ozone
        self.use_combined_textures = use_combined_textures
        self.use_half_precision = use_half_precision
        self.do_white_balance = do_white_balance
        self.use_luminance = use_luminance
        self.exposure = exposure
        self.planet_sea_level = planet_sea_level
        self.compute_shader = compute_shader
        self.material = material

        self.kBottomRadius = 6371000.0
        self.mesh = None
        self.m_model = None

    def start(self):
        inv_height_scale = 1.0 / self.planet.height_scale

        self.mesh = self.create_mesh()
        self.mesh.localScale = (1.0 / self.planet.scaled_space_factor) * self.top_bottom_radius_ratio * self.planet.radius * ((inv_height_scale + self.planet_sea_level) / inv_height_scale)
        self.mesh.material = self.material

        kLambdaMin = 360
        kLambdaMax = 830

        kSolarIrradiance = np.array([
            1.11776, 1.14259, 1.01249, 1.14716, 1.72765, 1.73054, 1.6887, 1.61253,
            1.91198, 2.03474, 2.02042, 2.02212, 1.93377, 1.95809, 1.91686, 1.8298,
            1.8685, 1.8931, 1.85149, 1.8504, 1.8341, 1.8345, 1.8147, 1.78158, 1.7533,
            1.6965, 1.68194, 1.64654, 1.6048, 1.52143, 1.55622, 1.5113, 1.474, 1.4482,
            1.41018, 1.36775, 1.34188, 1.31429, 1.28303, 1.26758, 1.2367, 1.2082,
            1.18737, 1.14683, 1.12362, 1.1058, 1.07124, 1.04992
        ])

        kOzoneCrossSection = np.array([
            1.18e-27, 2.182e-28, 2.818e-28, 6.636e-28, 1.527e-27, 2.763e-27, 5.52e-27,
            8.451e-27, 1.582e-26, 2.316e-26, 3.669e-26, 4.924e-26, 7.752e-26, 9.016e-26,
            1.48e-25, 1.602e-25, 2.139e-25, 2.755e-25, 3.091e-25, 3.5e-25, 4.266e-25,
            4.672e-25, 4.398e-25, 4.701e-25, 5.019e-25, 4.305e-25, 3.74e-25, 3.215e-25,
            2.662e-25, 2.238e-25, 1.852e-25, 1.473e-25, 1.209e-25, 9.423e-26, 7.455e-26,
            6.566e-26, 5.105e-26, 4.15e-26, 4.228e-26, 3.237e-26, 2.451e-26, 2.801e-26,
            2.534e-26, 1.624e-26, 1.465e-26, 2.078e-26, 1.383e-26, 7.105e-27
        ])

        kDobsonUnit = 2.687e20
        kMaxOzoneNumberDensity = 300.0 * kDobsonUnit / 15000.0
        kConstantSolarIrradiance = 1.5
        kTopRadius = self.kBottomRadius * self.top_bottom_radius_ratio
        kRayleigh = 1.24062e-6
        kRayleighScaleHeight = 8000.0
        kMieScaleHeight = 1200.0
        kMieAngstromAlpha = 0.0
        kMieAngstromBeta = 5.328e-3
        kMieSingleScatteringAlbedo = 0.9
        kMiePhaseFunctionG = 0.8
        kGroundAlbedo = 0.1
        max_sun_zenith_angle = (102.0 if self.use_half_precision else 120.0) / 180.0 * pi

        rayleigh_layer = DensityProfileLayer("rayleigh", 0.0, 1.0, -1.0 / kRayleighScaleHeight, 0.0, 0.0)
        mie_layer = DensityProfileLayer("mie", 0.0, 1.0, -1.0 / kMieScaleHeight, 0.0, 0.0)

        ozone_density = [
            DensityProfileLayer("absorption0", 25000.0, 0.0, 0.0, 1.0 / 15000.0, -2.0 / 3.0),
            DensityProfileLayer("absorption1", 0.0, 0.0, 0.0, -1.0 / 15000.0, 8.0 / 3.0)
        ]

        wavelengths = []
        solar_irradiance = []
        rayleigh_scattering = []
        mie_scattering = []
        mie_extinction = []
        absorption_extinction = []
        ground_albedo = []

        for l in range(kLambdaMin, kLambdaMax + 1, 10):
            lambda_um = l * 1e-3
            mie = kMieAngstromBeta / kMieScaleHeight * math.pow(lambda_um, -kMieAngstromAlpha)

            wavelengths.append(l)
            if self.use_constant_solar_spectrum:
                solar_irradiance.append(kConstantSolarIrradiance)
            else:
                solar_irradiance.append(kSolarIrradiance[(l - kLambdaMin) // 10])

            rayleigh_scattering.append(kRayleigh * lambda_um ** -4)
            mie_scattering.append(mie * kMieSingleScatteringAlbedo)
            mie_extinction.append(mie)
            absorption_extinction.append(kMaxOzoneNumberDensity * kOzoneCrossSection[(l - kLambdaMin) // 10] if self.use_ozone else 0.0)
            ground_albedo.append(kGroundAlbedo)

        self.m_model = Model()
        self.m_model.half_precision = self.use_half_precision
        self.m_model.combine_scattering_textures = self.use_combined_textures
        self.m_model.use_luminance = self.use_luminance
        self.m_model.wavelengths = wavelengths
        self.m_model.solar_irradiance = solar_irradiance
        self.m_model.sun_angular_radius = self.kSunAngularRadius
        self.m_model.bottom_radius = self.kBottomRadius
        self.m_model.top_radius = kTopRadius
        self.m_model.rayleigh_density = rayleigh_layer
        self.m_model.rayleigh_scattering = rayleigh_scattering
        self.m_model.mie_density = mie_layer
        self.m_model.mie_scattering = mie_scattering
        self.m_model.mie_extinction = mie_extinction
        self.m_model.mie_phase_function_g = kMiePhaseFunctionG
        self.m_model.absorption_density = ozone_density
        self.m_model.absorption_extinction = absorption_extinction
        self.m_model.ground_albedo = ground_albedo
        self.m_model.max_sun_zenith_angle = max_sun_zenith_angle
        self.m_model.length_unit_in_meters = self.kLengthUnitInMeters

        num_scattering_orders = 4
        self.m_model.init(self.compute_shader, num_scattering_orders)
        self.m_model.bind_to_material(self.material)

        self.set_up()

    def create_mesh(self):
        # Implement mesh creation
        pass

    def set_up(self):
        self.material.set_float("exposure", self.exposure * 1e-5 if self.use_luminance != "NONE" else self.exposure)
        self.material.set_vector("sun_size", (math.tan(self.kSunAngularRadius), math.cos(self.kSunAngularRadius)))

        white_point_r, white_point_g, white_point_b = 1.0, 1.0, 1.0
        if self.do_white_balance:
            white_point_r, white_point_g, white_point_b = self.m_model.convert_spectrum_to_linear_srgb()
            white_point = (white_point_r + white_point_g + white_point_b) / 3.0
            white_point_r /= white_point
            white_point_g /= white_point
            white_point_b /= white_point

        self.material.set_vector("white_point", (white_point_r, white_point_g, white_point_b))

    def on_destroy(self):
        if self.m_model is not None:
            self.m_model.release()

    def late_update(self):
        self.material.set_vector("earth_center", self.mesh.position * (self.top_bottom_radius_ratio / self.mesh.lossy_scale[0]))
        self.material.set_float("mesh_radius", 1.0 / self.mesh.lossy_scale[0])
        self.material.set_vector("sun_direction", self.sun.transform.forward * -1.0 if self.sun is not None else (0.0, 1.0, 0.0))

import math
import numpy as np

class ComputeShaderGenerator:
    xyz = "(x, y, z);"

    @staticmethod
    def generate_compute_shader(module):
        tree = []
        code = []

        try:
            ComputeShaderGenerator.build_tree(tree, module, 0)
        except Exception as e:
            print(f"Error: {e}")

        # Declaring heightmap textures and samplers
        for level in tree:
            for mod in level:
                if mod.op_type == 'Heightmap' or isinstance(mod, HeightmapModule):
                    hm = mod
                    if hm is None or hm.compute_shader_name is None:
                        continue
                    code.append(f"Texture2D<float> {hm.compute_shader_name};")
                    code.append(f"sampler sampler_{hm.compute_shader_name};")

        code.append("\nfloat get_noise(float x, float y, float z) {\n")

        input_ids = [""] * 3
        for i in range(len(tree) - 1, -1, -1):
            for j, mod in enumerate(tree[i]):
                if mod.inputs:
                    for k, inp in enumerate(mod.inputs):
                        for l in range(i, len(tree)):
                            try:
                                index = tree[l].index(inp)
                                input_ids[k] = ComputeShaderGenerator.id(l, index)
                                break
                            except ValueError:
                                pass

                if mod.op_type == 'Noise' or isinstance(mod, FastNoise):
                    n = mod
                    code.append(f"   set_parameters({n.get_seed()}, {n.frequency}, {n.octaves}, {n.lacunarity}, {int(n.fractal_type)});")
                    code.append(f"   float {ComputeShaderGenerator.id(i, j)}")

                    noise_types = {
                        'Value': " = get_value" + ComputeShaderGenerator.xyz,
                        'ValueFractal': " = get_value_fractal" + ComputeShaderGenerator.xyz,
                        'Perlin': " = get_perlin" + ComputeShaderGenerator.xyz,
                        'PerlinFractal': " = get_perlin_fractal" + ComputeShaderGenerator.xyz,
                        'Simplex': " = get_simplex" + ComputeShaderGenerator.xyz,
                        'SimplexFractal': " = get_simplex_fractal" + ComputeShaderGenerator.xyz,
                        'Cellular': " = get_cellular" + ComputeShaderGenerator.xyz,
                        'WhiteNoise': " = get_white_noise" + ComputeShaderGenerator.xyz,
                        'Cubic': " = get_cubic" + ComputeShaderGenerator.xyz,
                        'CubicFractal': " = get_cubic_fractal" + ComputeShaderGenerator.xyz
                    }
                    code.append(noise_types.get(n.noise_type, ""))

                elif mod.op_type == 'Heightmap' or isinstance(mod, HeightmapModule):
                    n = mod
                    id_n = ComputeShaderGenerator.id_n(i, j)
                    interpolation_method = "sample_bicubic" if n.use_bicubic_interpolation else "sample_linear"
                    code.append(f"   float f{id_n} = {interpolation_method}({n.compute_shader_name}, sampler_{n.compute_shader_name}, float3(x, y, z)) * 2.0 - 1.0;")
                else:
                    n = mod
                    id_n = ComputeShaderGenerator.id_n(i, j)

                    if n.op_type not in ['Curve', 'Terrace']:
                        code.append(f"   float f{id_n} = ")

                    module_operations = {
                        'Select': f"lerp({input_ids[1]}, {input_ids[0]}, select({input_ids[2]}, {n.parameters[2]:.6f}, {1 / (2 * n.parameters[0]):.6f}));",
                        'Curve': f"curve({input_ids[0]}, {n.curve.times}, {n.curve.values});",
                        'Blend': f"lerp({input_ids[0]}, {input_ids[1]}, {n.parameters[0]:.6f});",
                        'Remap': f"{input_ids[0]};",
                        'Add': f"{input_ids[0]} + {input_ids[1]};",
                        'Subtract': f"{input_ids[0]} - {input_ids[1]};",
                        'Multiply': f"{input_ids[0]} * {input_ids[1]};",
                        'Min': f"min({input_ids[0]}, {input_ids[1]});",
                        'Max': f"max({input_ids[0]}, {input_ids[1]});",
                        'Scale': f"{input_ids[0]} * {n.parameters[0]:.6f};",
                        'ScaleBias': f"mad({input_ids[0]}, {n.parameters[0]:.6f}, {n.parameters[1]:.6f});",
                        'Abs': f"abs({input_ids[0]});",
                        'Invert': f"-{input_ids[0]};",
                        'Clamp': f"clamp({input_ids[0]}, {n.parameters[0]:.6f}, {n.parameters[1]:.6f});",
                        'Const': f"{n.parameters[0]:.6f};",
                        'Terrace': f"terrace({input_ids[0]}, {n.parameters}, {len(n.parameters)});"
                    }
                    code.append(module_operations.get(n.op_type, ""))
                code.append("\n")

            if i == 0:
                code.append("   return (f0_0 + 1) / 2;\n")

        code.append("}\n")

        template = "your_compute_shader_template_here"  # Placeholder for actual template loading
        final_shader_code = template.replace("~", "".join(code))
        return final_shader_code

    @staticmethod
    def build_tree(tree, module, level):
        if len(tree) <= level:
            tree.append([])
        tree[level].append(module)
        if module.inputs:
            for inp in module.inputs:
                if not ComputeShaderGenerator.search(tree, level, inp):
                    ComputeShaderGenerator.build_tree(tree, inp, level + 1)

    @staticmethod
    def find_generators(module, ids, tree, level):
        if module.inputs:
            for inp in module.inputs:
                if inp.op_type == 'Noise':
                    ids.append(ComputeShaderGenerator.id(level, tree[level].index(inp)))
                else:
                    ComputeShaderGenerator.find_generators(inp, ids, tree, level + 1)

    @staticmethod
    def search(tree, current_level, module):
        if current_level < len(tree):
            for level in range(current_level, len(tree)):
                if module in tree[level]:
                    return True

            for level in range(current_level):
                try:
                    tree[level].remove(module)
                except ValueError:
                    pass

        return False

    @staticmethod
    def id(i, j):
        return f"f{i}_{j}"

    @staticmethod
    def id_n(i, j):
        return f"{i}_{j}"

import os
import re
import shutil
from pathlib import Path

class NodeEditorUtilities:

    script_icon = None  # Placeholder for the script icon equivalent in Python

    type_attributes = {}

    @staticmethod
    def get_attrib(cls, attrib_type):
        attribs = [attrib for attrib in cls.__dict__.values() if isinstance(attrib, attrib_type)]
        if attribs:
            return attribs[0]
        return None

    @staticmethod
    def get_attrib_from_instance(attribs, attrib_type):
        for attrib in attribs:
            if isinstance(attrib, attrib_type):
                return attrib
        return None

    @staticmethod
    def get_field_info(cls, field_name):
        for cls in cls.__mro__:
            if field_name in cls.__dict__:
                return cls.__dict__[field_name]
        return None

    @staticmethod
    def get_cached_attrib(cls, field_name, attrib_type):
        if cls not in NodeEditorUtilities.type_attributes:
            NodeEditorUtilities.type_attributes[cls] = {}

        if field_name not in NodeEditorUtilities.type_attributes[cls]:
            NodeEditorUtilities.type_attributes[cls][field_name] = {}

        if attrib_type not in NodeEditorUtilities.type_attributes[cls][field_name]:
            attrib = NodeEditorUtilities.get_attrib(cls, field_name, attrib_type)
            NodeEditorUtilities.type_attributes[cls][field_name][attrib_type] = attrib

        return NodeEditorUtilities.type_attributes[cls][field_name][attrib_type]

    @staticmethod
    def is_castable_to(from_type, to_type):
        return issubclass(from_type, to_type)

    @staticmethod
    def pretty_name(cls):
        if cls is None:
            return "null"
        if cls == object:
            return "object"
        if cls == float:
            return "float"
        if cls == int:
            return "int"
        if cls == bool:
            return "bool"
        if cls == str:
            return "string"
        if hasattr(cls, "__origin__") and cls.__origin__ is list:
            return f"List<{cls.__args__[0].__name__}>"
        if hasattr(cls, "__origin__") and cls.__origin__ is tuple:
            return f"Tuple<{', '.join(arg.__name__ for arg in cls.__args__)}>"
        return cls.__name__

    @staticmethod
    def create_node():
        NodeEditorUtilities.create_from_template("NewNode.py", "xNode_NodeTemplate.py")

    @staticmethod
    def create_graph():
        NodeEditorUtilities.create_from_template("NewNodeGraph.py", "xNode_NodeGraphTemplate.py")

    @staticmethod
    def create_from_template(initial_name, template_path):
        new_file_path = os.path.join(os.getcwd(), initial_name)
        shutil.copy(template_path, new_file_path)
        with open(new_file_path, 'r+') as file:
            content = file.read()
            content = re.sub(r'#SCRIPTNAME#', Path(initial_name).stem, content)
            file.seek(0)
            file.write(content)
            file.truncate()

        print(f"Created new file from template: {new_file_path}")

# Example usage:
# NodeEditorUtilities.create_node()
# NodeEditorUtilities.create_graph()

import json
import random
from collections import defaultdict

class Color:
    def __init__(self, r, g, b, a=255):
        self.r = r
        self.g = g
        self.b = b
        self.a = a

    def __repr__(self):
        return f"Color({self.r}, {self.g}, {self.b}, {self.a})"

    @staticmethod
    def from_hex(hex_str):
        hex_str = hex_str.lstrip('#')
        lv = len(hex_str)
        return Color(*(int(hex_str[i:i + lv // 4], 16) for i in range(0, lv, lv // 4)))

    def to_hex(self):
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}{self.a:02x}"


class Settings:
    def __init__(self):
        self.gridLineColor = Color(115, 115, 115)
        self.gridBgColor = Color(46, 46, 46)
        self.highlightColor = Color(255, 255, 255, 255)
        self.gridSnap = True
        self.autoSave = True
        self.zoomToMouse = True
        self.typeColorsData = ""
        self.typeColors = {}
        self.noodleType = "Curve"  # Assuming default value

    def to_dict(self):
        return {
            "gridLineColor": self.gridLineColor.to_hex(),
            "gridBgColor": self.gridBgColor.to_hex(),
            "highlightColor": self.highlightColor.to_hex(),
            "gridSnap": self.gridSnap,
            "autoSave": self.autoSave,
            "zoomToMouse": self.zoomToMouse,
            "typeColorsData": self.typeColorsData,
            "typeColors": {k: v.to_hex() for k, v in self.typeColors.items()},
            "noodleType": self.noodleType,
        }

    @classmethod
    def from_dict(cls, data):
        instance = cls()
        instance.gridLineColor = Color.from_hex(data["gridLineColor"])
        instance.gridBgColor = Color.from_hex(data["gridBgColor"])
        instance.highlightColor = Color.from_hex(data["highlightColor"])
        instance.gridSnap = data["gridSnap"]
        instance.autoSave = data["autoSave"]
        instance.zoomToMouse = data["zoomToMouse"]
        instance.typeColorsData = data["typeColorsData"]
        instance.typeColors = {k: Color.from_hex(v) for k, v in data["typeColors"].items()}
        instance.noodleType = data["noodleType"]
        return instance


class NodeEditorPreferences:
    NoodleType = ["Curve", "Line", "Angled"]

    lastEditor = None
    lastKey = "xNode.Settings"
    typeColors = defaultdict(lambda: Color(128, 128, 128))
    settings = {}

    @staticmethod
    def get_settings():
        if NodeEditorPreferences.lastEditor != NodeEditorWindow.current.graphEditor:
            attribs = getattr(NodeEditorWindow.current.graphEditor, "__annotations__", {})
            if "CustomNodeGraphEditorAttribute" in attribs:
                NodeEditorPreferences.lastEditor = NodeEditorWindow.current.graphEditor
                NodeEditorPreferences.lastKey = attribs["CustomNodeGraphEditorAttribute"].editorPrefsKey
            else:
                return None
        if NodeEditorPreferences.lastKey not in NodeEditorPreferences.settings:
            NodeEditorPreferences.verify_loaded()
        return NodeEditorPreferences.settings[NodeEditorPreferences.lastKey]

    @staticmethod
    def preferences_gui():
        NodeEditorPreferences.verify_loaded()
        settings = NodeEditorPreferences.settings[NodeEditorPreferences.lastKey]

        NodeEditorPreferences.node_settings_gui(NodeEditorPreferences.lastKey, settings)
        NodeEditorPreferences.grid_settings_gui(NodeEditorPreferences.lastKey, settings)
        NodeEditorPreferences.system_settings_gui(NodeEditorPreferences.lastKey, settings)
        NodeEditorPreferences.type_colors_gui(NodeEditorPreferences.lastKey, settings)
        if input("Press 'd' to set defaults: ").lower() == 'd':
            NodeEditorPreferences.reset_prefs()

    @staticmethod
    def grid_settings_gui(key, settings):
        print("Grid Settings")
        settings.gridSnap = input("Snap (y/n): ").strip().lower() == 'y'
        settings.zoomToMouse = input("Zoom to Mouse (y/n): ").strip().lower() == 'y'

        settings.gridLineColor = Color.from_hex(input("Grid Line Color (hex): ").strip())
        settings.gridBgColor = Color.from_hex(input("Grid BG Color (hex): ").strip())
        NodeEditorPreferences.save_prefs(key, settings)

    @staticmethod
    def system_settings_gui(key, settings):
        print("System Settings")
        settings.autoSave = input("Auto Save (y/n): ").strip().lower() == 'y'
        NodeEditorPreferences.save_prefs(key, settings)

    @staticmethod
    def node_settings_gui(key, settings):
        print("Node Settings")
        settings.highlightColor = Color.from_hex(input("Highlight Color (hex): ").strip())
        settings.noodleType = input("Noodle Type (Curve/Line/Angled): ").strip()
        NodeEditorPreferences.save_prefs(key, settings)

    @staticmethod
    def type_colors_gui(key, settings):
        print("Type Colors")
        for type_name, color in NodeEditorPreferences.typeColors.items():
            new_color = Color.from_hex(input(f"{type_name} Color (hex): ").strip())
            NodeEditorPreferences.typeColors[type_name] = new_color
            settings.typeColors[type_name] = new_color
            NodeEditorPreferences.save_prefs(key, settings)

    @staticmethod
    def load_prefs():
        if not os.path.exists(NodeEditorPreferences.lastKey):
            return Settings()
        with open(NodeEditorPreferences.lastKey, 'r') as file:
            data = json.load(file)
        return Settings.from_dict(data)

    @staticmethod
    def reset_prefs():
        if os.path.exists(NodeEditorPreferences.lastKey):
            os.remove(NodeEditorPreferences.lastKey)
        if NodeEditorPreferences.lastKey in NodeEditorPreferences.settings:
            del NodeEditorPreferences.settings[NodeEditorPreferences.lastKey]
        NodeEditorPreferences.typeColors = defaultdict(lambda: Color(128, 128, 128))
        NodeEditorPreferences.verify_loaded()

    @staticmethod
    def save_prefs(key, settings):
        with open(key, 'w') as file:
            json.dump(settings.to_dict(), file)

    @staticmethod
    def verify_loaded():
        if NodeEditorPreferences.lastKey not in NodeEditorPreferences.settings:
            NodeEditorPreferences.settings[NodeEditorPreferences.lastKey] = NodeEditorPreferences.load_prefs()

    @staticmethod
    def get_type_color(type_):
        NodeEditorPreferences.verify_loaded()
        if type_ is None or type_ == 'ModuleWrapper':
            return Color(128, 128, 128)
        if type_ not in NodeEditorPreferences.typeColors:
            type_name = type_
            if type_name in NodeEditorPreferences.settings[NodeEditorPreferences.lastKey].typeColors:
                NodeEditorPreferences.typeColors[type_] = NodeEditorPreferences.settings[NodeEditorPreferences.lastKey].typeColors[type_name]
            else:
                random.seed(type_name)
                color = Color(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                NodeEditorPreferences.typeColors[type_] = color
        return NodeEditorPreferences.typeColors[type_]


class NodeEditorWindow:
    current = None

    @staticmethod
    def repaint_all():
        print("Repainting all Node Editor windows")

class NodeGraphEditor:
    pass

import json
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Union

class ShowBackingValue(Enum):
    Never = 0
    Unconnected = 1
    Always = 2

class ConnectionType(Enum):
    Multiple = 0
    Override = 1


class TypeConstraint(Enum):
    NoConstraint = 0  # Renamed 'None' to 'NoConstraint'
    Inherited = 1
    Strict = 2


class NodePort:
    class IO(Enum):
        Input = 0
        Output = 1

    def __init__(self, field_name, node_type, direction, connection_type, type_constraint, owner):
        self.field_name = field_name
        self.node_type = node_type
        self.direction = direction
        self.connection_type = connection_type
        self.type_constraint = type_constraint
        self.owner = owner
        self.connections = []

    def is_output(self):
        return self.direction == NodePort.IO.Output

    def is_input(self):
        return self.direction == NodePort.IO.Input

    def is_dynamic(self):
        return False

    def is_connected(self):
        return len(self.connections) > 0

    def get_input_value(self):
        # Placeholder for actual input value retrieval logic
        return None

    def get_input_values(self):
        # Placeholder for actual input values retrieval logic
        return []

    def verify_connections(self):
        # Placeholder for connection verification logic
        pass

    def clear_connections(self):
        self.connections.clear()

class NodeGraph:
    pass

class Vector2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Node:
    class InputAttribute:
        def __init__(self, backing_value=ShowBackingValue.Unconnected, connection_type=ConnectionType.Multiple, type_constraint=TypeConstraint.NoConstraint, instance_port_list=False):
            self.backing_value = backing_value
            self.connection_type = connection_type
            self.type_constraint = type_constraint
            self.instance_port_list = instance_port_list
    class OutputAttribute:
        def __init__(self, backing_value=ShowBackingValue.Never, connection_type=ConnectionType.Multiple, instance_port_list=False):
            self.backing_value = backing_value
            self.connection_type = connection_type
            self.instance_port_list = instance_port_list

    class CreateNodeMenuAttribute:
        def __init__(self, menu_name):
            self.menu_name = menu_name

    class NodeTintAttribute:
        def __init__(self, r, g, b):
            self.color = (r, g, b)

    class NodeWidthAttribute:
        def __init__(self, width):
            self.width = width

    graph_hotfix = None

    def __init__(self):
        self.ports = Node.NodePortDictionary()
        self.graph = None
        self.position = Vector2(0, 0)
        if Node.graph_hotfix:
            self.graph = Node.graph_hotfix
        Node.graph_hotfix = None
        self.update_static_ports()
        self.init()

    def on_enable(self):
        if Node.graph_hotfix:
            self.graph = Node.graph_hotfix
        Node.graph_hotfix = None
        self.update_static_ports()
        self.init()

    def update_static_ports(self):
        # Placeholder for updating static ports logic
        pass

    def init(self):
        # Placeholder for node initialization logic
        pass

    def verify_connections(self):
        for port in self.ports.values():
            port.verify_connections()

    def add_instance_input(self, node_type, connection_type=ConnectionType.Multiple, type_constraint=TypeConstraint.NoConstraint, field_name=None):
        return self.add_instance_port(node_type, NodePort.IO.Input, connection_type, type_constraint, field_name)

    def add_instance_output(self, node_type, connection_type=ConnectionType.Multiple, type_constraint=TypeConstraint.NoConstraint, field_name=None):
        return self.add_instance_port(node_type, NodePort.IO.Output, connection_type, type_constraint, field_name)

from enum import Enum

class TypeConstraint(Enum):
    NoConstraint = 0  # Renamed 'None' to 'NoConstraint'
    Inherited = 1
    Strict = 2

class ConnectionType(Enum):
    Multiple = 0
    Single = 1

class ShowBackingValue(Enum):
    Unconnected = 0
    Connected = 1

class NodePort:
    def __init__(self, field_name, node_type, direction, connection_type, type_constraint, parent):
        self.field_name = field_name
        self.node_type = node_type
        self.direction = direction
        self.connection_type = connection_type
        self.type_constraint = type_constraint
        self.parent = parent
        self.ports = {}
        self.name = "NodePort"
        self.instance_ports = []
        
    def is_output(self):
        return self.direction == 'Output'

    def is_input(self):
        return self.direction == 'Input'

    def is_connected(self):
        return True  # Placeholder implementation

    def get_input_value(self):
        return None  # Placeholder implementation

    def get_input_values(self):
        return None  # Placeholder implementation

    def is_dynamic(self):
        return True  # Placeholder implementation

    def clear_connections(self):
        pass  # Placeholder implementation

    def add_instance_port(self, node_type, direction, connection_type=ConnectionType.Multiple, type_constraint=TypeConstraint.NoConstraint, field_name=None):
        if field_name is None:
            field_name = "instanceInput_0"
            i = 0
            while self.has_port(field_name):
                field_name = f"instanceInput_{i}"
                i += 1
        elif self.has_port(field_name):
            print(f"Port '{field_name}' already exists in {self.name}")
            return self.ports[field_name]
        port = NodePort(field_name, node_type, direction, connection_type, type_constraint, self)
        self.ports[field_name] = port
        return port

    def remove_instance_port(self, field_name):
        self.remove_instance_port(self.get_port(field_name))

    def remove_instance_port(self, port):
        if port is None:
            raise ValueError("port cannot be None")
        if not port.is_dynamic():
            raise ValueError("cannot remove static port")
        port.clear_connections()
        del self.ports[port.field_name]

    def clear_instance_ports(self):
        instance_ports = list(self.instance_ports)
        for port in instance_ports:
            self.remove_instance_port(port)

    def get_output_port(self, field_name):
        port = self.get_port(field_name)
        if port and port.is_output():
            return port
        return None

    def get_input_port(self, field_name):
        port = self.get_port(field_name)
        if port and port.is_input():
            return port
        return None

    def get_port(self, field_name):
        return self.ports.get(field_name, None)

    def has_port(self, field_name):
        return field_name in self.ports

    def get_input_value(self, field_name, fallback=None):
        port = self.get_port(field_name)
        if port and port.is_connected():
            return port.get_input_value()
        return fallback

    def get_input_values(self, field_name, fallback=None):
        port = self.get_port(field_name)
        if port and port.is_connected():
            return port.get_input_values()
        return fallback

    def get_value(self, port):
        print(f"No get_value(port) override defined for {type(self).__name__}")
        return None


    def on_create_connection(self, from_port, to_port):
        pass

    def on_remove_connection(self, port):
        pass

    def clear_connections(self):
        for port in self.ports.values():
            port.clear_connections()

    class NodePortDictionary(dict):
        def on_before_serialize(self):
            keys = list(self.keys())
            values = list(self.values())
            return keys, values

        def on_after_deserialize(self, keys, values):
            self.clear()
            if len(keys) != len(values):
                raise Exception(f"there are {len(keys)} keys and {len(values)} values after deserialization. Make sure that both key and value types are serializable.")
            for i in range(len(keys)):
                self[keys[i]] = values[i]

import numpy as np

class Vector2:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

class Vector3:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def normalize(self):
        norm = np.linalg.norm([self.x, self.y, self.z])
        self.x /= norm
        self.y /= norm
        self.z /= norm
        return self

    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar):
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def magnitude(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

class Color32:
    def __init__(self, r=0, g=0, b=0, a=255):
        self.r = r
        self.g = g
        self.b = b
        self.a = a

    @staticmethod
    def lerp(color1, color2, t):
        r = int(color1.r + (color2.r - color1.r) * t)
        g = int(color1.g + (color2.g - color1.g) * t)
        b = int(color1.b + (color2.b - color1.b) * t)
        a = int(color1.a + (color2.a - color1.a) * t)
        return Color32(r, g, b, a)

class MeshData:
    def __init__(self, vertices, colors=None, normals=None, uv=None, uv2=None):
        self.vertices = vertices
        self.colors = colors if colors is not None else [Color32() for _ in vertices]
        self.normals = normals if normals is not None else [Vector3() for _ in vertices]
        self.uv = uv if uv is not None else [Vector2() for _ in vertices]
        self.uv2 = uv2 if uv2 is not None else [Vector2() for _ in vertices]

class MeshGeneration:
    @staticmethod
    def generate_mesh(quad, md):
        planet = quad.planet
        side_length = planet.quad_size
        side_length_1 = side_length - 1
        num_vertices = side_length * side_length

        final_verts = [Vector3() for _ in range(num_vertices)]
        height = 0
        down = Vector3()
        normalized = Vector3()
        average_height = 0
        heights = [0 for _ in range(num_vertices)] if planet.calculate_msds else None

        offsetX, offsetY, level_constant = 0, 0, 0
        MeshGeneration.calculate_uv_constants(quad, level_constant, offsetX, offsetY)

        md.uv = [Vector2() for _ in range(num_vertices)]
        for i in range(num_vertices):
            md.vertices[i], height, normalized, md.uv[i] = MeshGeneration.get_position(quad, md.vertices[i])
            if planet.calculate_msds:
                average_height += height
                heights[i] = height
            if i == 0:
                quad.mesh_offset = md.vertices[0]
                down = (Vector3() - quad.mesh_offset).normalize()

            md.vertices[i] -= quad.mesh_offset
            final_verts[i] = md.vertices[i]

            if not planet.using_legacy_uv_type:
                if planet.uv_type == "Cube":
                    MeshGeneration.calculate_uv_cube(quad, md.uv[i], i, side_length, side_length_1, planet.uv_scale, level_constant, offsetX, offsetY)
                else:
                    MeshGeneration.calculate_uv_quad(quad, md.uv[i], i, side_length, side_length_1, level_constant)

            texture = planet.texture_provider.evaluate_texture(height, normalized)
            md.colors[i] = Color32(texture[0], texture[1], texture[2], texture[3])
            md.uv2[i] = Vector2(texture[4], texture[5])

            for j in range(len(texture)):
                if texture[j] > 0.5:
                    quad.biome |= (1 << j)

        for i in range(num_vertices, len(md.vertices)):
            md.vertices[i] = MeshGeneration.get_position(quad, md.vertices[i])
            md.vertices[i] -= quad.mesh_offset

        MeshGeneration.calculate_normals(md.normals, md.vertices, planet.quad_arrays.tris_extended_plane, num_vertices)
        md.vertices = final_verts

        MeshGeneration.slope_texture(planet, md, num_vertices, down)

        if planet.calculate_msds:
            average_height /= num_vertices
            for i in range(num_vertices):
                deviation = average_height - heights[i]
                quad.msd += deviation * deviation

        return md

    @staticmethod
    def calculate_uv_constants(quad, level_constant, offsetX, offsetY):
        planet = quad.planet
        if planet.uv_type == "Quad":
            level_constant = 1 << (len(planet.detail_distances) - quad.level)
        elif planet.uv_type == "Cube":
            level_constant = 1 << quad.level
            indices = quad.index
            p = 0.5 * planet.uv_scale
            for idx in indices[2:]:
                if idx == 3:
                    offsetY += p
                elif idx == 1:
                    offsetX += p
                    offsetY += p
                elif idx == 2:
                    pass
                elif idx == 0:
                    offsetX += p
                p *= 0.5
            offsetX %= 1
            offsetY %= 1

    @staticmethod
    def get_position(quad, vertex):
        planet = quad.planet
        vertex = vertex * quad.scale
        vertex = quad.rotation * vertex
        if planet.using_legacy_uv_type:
            if planet.uv_type == "LegacyContinuous":
                vertex += quad.tr_position
            if quad.plane == "ZPlane":
                uv = Vector2(vertex.x, vertex.y)
            elif quad.plane == "YPlane":
                uv = Vector2(vertex.x, vertex.z)
            elif quad.plane == "XPlane":
                uv = Vector2(vertex.z, vertex.y)
            else:
                uv = Vector2()
            if planet.uv_type != "LegacyContinuous":
                vertex += quad.tr_position
        else:
            vertex += quad.tr_position
            uv = Vector2()

        vertex.normalize()
        height = planet.height_provider.height_at_xyz(vertex)
        vertex *= planet.radius
        vertex -= quad.tr_position
        vertex *= (planet.height_inv + height) / planet.height_inv
        return vertex, height, vertex, uv

    @staticmethod
    def calculate_uv_cube(quad, uv, i, side_length, side_length_1, uv_scale, level_constant, offsetX, offsetY):
        x = (i // side_length) / side_length_1
        y = (i % side_length) / side_length_1
        scale = uv_scale / level_constant
        x *= scale
        y *= scale
        x += offsetX
        y += offsetY
        y *= -1
        MeshGeneration.rotate_uv(quad, x, y)
        uv.x, uv.y = x, y

    @staticmethod
    def calculate_uv_quad(quad, uv, i, side_length, side_length_1, level_constant):
        x = (i // side_length) / side_length_1
        y = (i % side_length) / side_length_1
        x *= level_constant
        y *= level_constant
        y = -y
        MeshGeneration.rotate_uv(quad, x, y)
        uv.x, uv.y = x, y

    @staticmethod
    def rotate_uv(quad, x, y):
        if quad.position == "Front":
            if quad.plane == "YPlane":
                return
            elif quad.plane == "ZPlane":
                return
            elif quad.plane == "XPlane":
                x, y = y, -x
                return
        else:
            if quad.plane == "YPlane":
                x, y = -x, -y
                return
            elif quad.plane == "ZPlane":
                x, y = -x, -y
                return
            elif quad.plane == "XPlane":
                x, y = -y, x
                return

    @staticmethod
    def calculate_normals(normals, vertices, tris, num_vertices):
        len_normals = len(normals)
        for i in range(0, len(tris), 3):
            p1 = vertices[tris[i]]
            p2 = vertices[tris[i + 1]]
            p3 = vertices[tris[i + 2]]
            l1 = p2 - p1
            l2 = p3 - p1
            normal = l1.cross(l2)
            for n in [tris[i], tris[i + 1], tris[i + 2]]:
                if n < len_normals:
                    normals[n] = normals[n] + normal

        for i in range(len_normals):
            length = normals[i].magnitude()
            normals[i] = normals[i] / length

    @staticmethod
    def slope_texture(planet, md, num_vertices, down):
        if planet.slope_texture_type == "Fade":
            for i in range(num_vertices):
                slope = np.degrees(np.arccos(down.dot(md.normals[i])))
                if slope > planet.slope_angle - planet.slope_fade_in_angle:
                    fade = np.clip((slope - planet.slope_fade_in_angle) / (planet.slope_angle - planet.slope_fade_in_angle), 0, 1)
                    texture = [0, 0, 0, 0, 0, 0]
                    texture[planet.slope_texture] = 1.0
                    md.colors[i] = Color32.lerp(md.colors[i], Color32(*texture[:4]), fade)
                    md.uv2[i] = Vector2.lerp(md.uv2[i], Vector2(*texture[4:]), fade)
        elif planet.slope_texture_type == "Threshold":
            slope_angle = np.radians(planet.slope_angle)
            for i in range(num_vertices):
                if np.arccos(down.dot(md.normals[i])) > slope_angle:
                    texture = [0, 0, 0, 0, 0, 0]
                    texture[planet.slope_texture] = 1.0
                    md.colors[i] = Color32(*texture[:4])
                    md.uv2[i] = Vector2(*texture[4:])

import json
from typing import List, Type

class Node:
    class ConnectionType:
        Multiple = "Multiple"
        Override = "Override"

    class TypeConstraint:
        NoneType = "None"
        Inherited = "Inherited"
        Strict = "Strict"

    def __init__(self):
        self.ports = {}

    def GetPort(self, fieldName: str):
        return self.ports.get(fieldName)

    def GetValue(self, port):
        pass

    def OnCreateConnection(self, from_port, to_port):
        pass

    def OnRemoveConnection(self, port):
        pass

class NodePort:
    class IO:
        Input = "Input"
        Output = "Output"

    def __init__(self, fieldName: str, node: Node, valueType: Type, direction: str, connectionType: str, typeConstraint: str, dynamic: bool):
        self._fieldName = fieldName
        self._node = node
        self.valueType = valueType
        self._direction = direction
        self._connectionType = connectionType
        self._typeConstraint = typeConstraint
        self._dynamic = dynamic
        self.connections = []

    @property
    def ConnectionCount(self):
        return len(self.connections)

    @property
    def Connection(self):
        for conn in self.connections:
            if conn.Port is not None:
                return conn.Port
        return None

    @property
    def direction(self):
        return self._direction

    @property
    def connectionType(self):
        return self._connectionType

    @property
    def typeConstraint(self):
        return self._typeConstraint

    @property
    def IsConnected(self):
        return len(self.connections) != 0

    @property
    def IsInput(self):
        return self.direction == self.IO.Input

    @property
    def IsOutput(self):
        return self.direction == self.IO.Output

    @property
    def fieldName(self):
        return self._fieldName

    @property
    def node(self):
        return self._node

    @property
    def IsDynamic(self):
        return self._dynamic

    @property
    def IsStatic(self):
        return not self._dynamic

    def VerifyConnections(self):
        self.connections = [conn for conn in self.connections if conn.node and conn.fieldName and conn.node.GetPort(conn.fieldName) is not None]

    def GetOutputValue(self):
        if self.direction == self.IO.Input:
            return None
        return self.node.GetValue(self)

    def GetInputValue(self):
        connectedPort = self.Connection
        if connectedPort is None:
            return None
        return connectedPort.GetOutputValue()

    def GetInputValues(self):
        return [conn.Port.GetOutputValue() for conn in self.connections if conn.Port is not None]

    def GetInputValueT(self, fallback=None):
        value = self.GetInputValue()
        return value if isinstance(value, type(fallback)) else fallback

    def GetInputValuesT(self):
        return [value for value in self.GetInputValues() if isinstance(value, type(fallback))]

    def TryGetInputValue(self, fallback=None):
        value = self.GetInputValue()
        if isinstance(value, type(fallback)):
            return value, True
        else:
            return fallback, False

    def GetInputSum(self, fallback):
        values = self.GetInputValues()
        if not values:
            return fallback
        return sum(values)

    def Connect(self, port):
        if not port:
            print("Cannot connect to null port")
            return
        if port == self:
            print("Cannot connect port to self.")
            return
        if self.IsConnectedTo(port):
            print("Port already connected.")
            return
        if self.direction == port.direction:
            print(f"Cannot connect two {self.direction} connections")
            return
        if port.connectionType == Node.ConnectionType.Override and port.ConnectionCount != 0:
            port.ClearConnections()
        if self.connectionType == Node.ConnectionType.Override and self.ConnectionCount != 0:
            self.ClearConnections()
        self.connections.append(PortConnection(port))
        if not port.IsConnectedTo(self):
            port.connections.append(PortConnection(self))
        self.node.OnCreateConnection(self, port)
        port.node.OnCreateConnection(self, port)

    def GetConnections(self):
        return [conn.Port for conn in self.connections if conn.Port is not None]

    def GetConnection(self, i):
        conn = self.connections[i]
        if conn.node is None or not conn.fieldName:
            self.connections.pop(i)
            return None
        port = conn.node.GetPort(conn.fieldName)
        if port is None:
            self.connections.pop(i)
            return None
        return port

    def GetConnectionIndex(self, port):
        for i, conn in enumerate(self.connections):
            if conn.Port == port:
                return i
        return -1

    def IsConnectedTo(self, port):
        return any(conn.Port == port for conn in self.connections)

    def CanConnectTo(self, port):
        if self.IsInput:
            input_port = self
            output_port = port
        else:
            input_port = port
            output_port = self

        if input_port is None or output_port is None:
            return False

        if input_port.typeConstraint == Node.TypeConstraint.Inherited and not issubclass(output_port.valueType, input_port.valueType):
            return False
        if input_port.typeConstraint == Node.TypeConstraint.Strict and input_port.valueType != output_port.valueType:
            return False

        return True

    def Disconnect(self, port):
        self.connections = [conn for conn in self.connections if conn.Port != port]
        if port is not None:
            port.connections = [conn for conn in port.connections if conn.Port != self]
        self.node.OnRemoveConnection(self)
        if port is not None:
            port.node.OnRemoveConnection(port)

    def DisconnectByIndex(self, i):
        other_port = self.connections[i].Port
        if other_port is not None:
            other_port.connections = [conn for conn in other_port.connections if conn.Port != self]
        self.connections.pop(i)
        self.node.OnRemoveConnection(self)
        if other_port is not None:
            other_port.node.OnRemoveConnection(other_port)

    def ClearConnections(self):
        while self.connections:
            self.Disconnect(self.connections[0].Port)

    def GetReroutePoints(self, index):
        return self.connections[index].reroutePoints

    def SwapConnections(self, target_port):
        port_connections = self.GetConnections()
        target_port_connections = target_port.GetConnections()
        self.ClearConnections()
        target_port.ClearConnections()

        for conn in port_connections:
            target_port.Connect(conn)

        for conn in target_port_connections:
            self.Connect(conn)

    def AddConnections(self, target_port):
        for conn in target_port.connections:
            other_port = conn.Port
            self.Connect(other_port)

    def MoveConnections(self, target_port):
        for conn in self.connections:
            other_port = conn.Port
            target_port.Connect(other_port)
        self.ClearConnections()

    def Redirect(self, old_nodes, new_nodes):
        for conn in self.connections:
            index = old_nodes.index(conn.node)
            if index >= 0:
                conn.node = new_nodes[index]

class PortConnection:
    def __init__(self, port):
        self.fieldName = port.fieldName
        self.node = port.node
        self.port = port
        self.reroutePoints = []

    @property
    def Port(self):
        return self.port

import math
from typing import List, Type
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

class NodeGraphEditor:
    def __init__(self, graph):
        self.graph = graph
        self.position = (0, 0)

    def GetEditor(self, graph):
        return self

    def OnGUI(self):
        pass

    def GetGridTexture(self):
        # Return a grid texture for drawing
        pass

    def GetSecondaryGridTexture(self):
        # Return a secondary grid texture for drawing
        pass

    def GetTypeColor(self, valueType):
        return (1, 1, 1, 1)  # White color as placeholder

class NodeEditorPreferences:
    class NoodleType:
        Curve = "Curve"
        Line = "Line"
        Angled = "Angled"

    @staticmethod
    def GetSettings():
        return NodeEditorPreferences()

    def __init__(self):
        self.noodleType = NodeEditorPreferences.NoodleType.Curve
        self.highlightColor = (1, 1, 1, 1)  # White color as placeholder
        self.autoSave = False

class NodeEditorWindow:
    def __init__(self):
        self.graph = None
        self.graphEditor = None
        self.selectionCache = []
        self.culledNodes = []
        self.onLateGUI = None
        self.zoom = 1.0
        self.panOffset = (0, 0)
        self.currentActivity = None
        self.dragBoxStart = (0, 0)
        self.preBoxSelectionReroute = None
        self.selectedReroutes = []
        self.portConnectionPoints = {}
        self.nodeSizes = {}
        self.hoveredNode = None
        self.hoveredPort = None
        self.draggedConnection = None

    @property
    def topPadding(self):
        return 19 if self.isDocked() else 22

    def isDocked(self):
        # Placeholder for docked check
        return False

    def OnGUI(self):
        if self.graph is None:
            return

        self.graphEditor = NodeGraphEditor(self.graph)
        self.graphEditor.position = self.position

        self.Controls()

        self.DrawGrid(self.position, self.zoom, self.panOffset)
        self.DrawConnections()
        self.DrawDraggedConnection()
        self.DrawNodes()
        self.DrawSelectionBox()
        self.DrawTooltip()
        self.graphEditor.OnGUI()

        if self.onLateGUI:
            self.onLateGUI()
            self.onLateGUI = None

    def Controls(self):
        # Placeholder for controls handling
        pass

    @staticmethod
    def BeginZoomed(rect, zoom, topPadding):
        plt.cla()
        plt.xlim(rect[0] * zoom, rect[2] * zoom)
        plt.ylim(rect[1] * zoom, rect[3] * zoom)

    @staticmethod
    def EndZoomed(rect, zoom, topPadding):
        plt.xlim(rect[0], rect[2])
        plt.ylim(rect[1], rect[3])

    def DrawGrid(self, rect, zoom, panOffset):
        plt.grid(True)

    def DrawSelectionBox(self):
        if self.currentActivity == "DragGrid":
            curPos = self.WindowToGridPosition(plt.ginput(1)[0])
            size = (curPos[0] - self.dragBoxStart[0], curPos[1] - self.dragBoxStart[1])
            r = Rectangle(self.dragBoxStart, size[0], size[1], linewidth=1, edgecolor='r', facecolor='none')
            plt.gca().add_patch(r)

    @staticmethod
    def DropdownButton(name, width):
        # Placeholder for dropdown button
        return False

    def ShowRerouteContextMenu(self, reroute):
        # Placeholder for reroute context menu
        pass

    def ShowPortContextMenu(self, hoveredPort):
        # Placeholder for port context menu
        pass

    def DrawConnection(self, startPoint, endPoint, col):
        startPoint = self.GridToWindowPosition(startPoint)
        endPoint = self.GridToWindowPosition(endPoint)
        plt.plot([startPoint[0], endPoint[0]], [startPoint[1], endPoint[1]], color=col)

    def DrawConnections(self):
        for node in self.graph.nodes:
            if node is None:
                continue
            for output in node.Outputs:
                if output not in self.portConnectionPoints:
                    continue
                connectionColor = self.graphEditor.GetTypeColor(output.ValueType)
                for k in range(output.ConnectionCount):
                    input_port = output.GetConnection(k)
                    if input_port is None:
                        continue
                    if not input_port.IsConnectedTo(output):
                        input_port.Connect(output)
                    if input_port not in self.portConnectionPoints:
                        continue

                    from_point = self.portConnectionPoints[output].center
                    to_point = self.portConnectionPoints[input_port].center
                    self.DrawConnection(from_point, to_point, connectionColor)

    def DrawNodes(self):
        # Placeholder for drawing nodes
        pass

    def DrawTooltip(self):
        if self.hoveredPort:
            type_name = type(self.hoveredPort.ValueType).__name__
            plt.text(plt.ginput(1)[0][0], plt.ginput(1)[0][1], type_name)

    def GridToWindowPosition(self, point):
        return (point[0] * self.zoom + self.panOffset[0], point[1] * self.zoom + self.panOffset[1])

    def WindowToGridPosition(self, point):
        return ((point[0] - self.panOffset[0]) / self.zoom, (point[1] - self.panOffset[1]) / self.zoom)

# Example usage
fig = plt.figure()
ax = fig.add_subplot(111)
editor_window = NodeEditorWindow()
editor_window.OnGUI()
plt.show()

import json
from typing import Any, Dict, List, Callable, Type
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class NodeEditorGUILayout:
    reorderableListCache: Dict[Any, Dict[str, Any]] = {}
    reorderableListIndex: int = -1

    @staticmethod
    def PropertyField(property: Any, includeChildren: bool = True, *options: Any) -> None:
        NodeEditorGUILayout.PropertyFieldWithLabel(property, None, includeChildren, *options)

    @staticmethod
    def PropertyFieldWithLabel(property: Any, label: Any, includeChildren: bool = True, *options: Any) -> None:
        if property is None:
            raise ValueError("Property cannot be None")

        node = property.serializedObject.targetObject
        port = node.GetPort(property.name)
        NodeEditorGUILayout.PropertyFieldWithPort(property, label, port, includeChildren, *options)

    @staticmethod
    def PropertyFieldWithPort(property: Any, port: Any, includeChildren: bool = True, *options: Any) -> None:
        NodeEditorGUILayout.PropertyFieldWithPortAndLabel(property, None, port, includeChildren, *options)

    @staticmethod
    def PropertyFieldWithPortAndLabel(property: Any, label: Any, port: Any, includeChildren: bool = True, *options: Any) -> None:
        if property is None:
            raise ValueError("Property cannot be None")

        if port is None:
            NodeEditorGUILayout.DrawPropertyField(property, label, includeChildren, *options)
        else:
            NodeEditorGUILayout.DrawPortField(property, label, port, includeChildren)

    @staticmethod
    def DrawPropertyField(property: Any, label: Any, includeChildren: bool, *options: Any) -> None:
        # Implement a method to draw a property field similar to EditorGUILayout.PropertyField in Unity
        # Placeholder implementation for example purposes
        print(f"Drawing property field: {property.name}")

    @staticmethod
    def DrawPortField(property: Any, label: Any, port: Any, includeChildren: bool) -> None:
        rect = Rectangle((0, 0), 16, 16)  # Placeholder for the position of the port handle
        plt.gca().add_patch(rect)

        if port.direction == "Input":
            NodeEditorGUILayout.DrawInputPort(property, label, port, includeChildren)
        elif port.direction == "Output":
            NodeEditorGUILayout.DrawOutputPort(property, label, port, includeChildren)

    @staticmethod
    def DrawInputPort(property: Any, label: Any, port: Any, includeChildren: bool) -> None:
        showBacking = port.showBacking

        if showBacking == "Unconnected":
            if port.IsConnected:
                NodeEditorGUILayout.LabelField(label or property.displayName)
            else:
                NodeEditorGUILayout.DrawPropertyField(property, label, includeChildren)
        elif showBacking == "Never":
            NodeEditorGUILayout.LabelField(label or property.displayName)
        elif showBacking == "Always":
            NodeEditorGUILayout.DrawPropertyField(property, label, includeChildren)

    @staticmethod
    def DrawOutputPort(property: Any, label: Any, port: Any, includeChildren: bool) -> None:
        showBacking = port.showBacking

        if showBacking == "Unconnected":
            if port.IsConnected:
                NodeEditorGUILayout.LabelField(label or property.displayName)
            else:
                NodeEditorGUILayout.DrawPropertyField(property, label, includeChildren)
        elif showBacking == "Never":
            NodeEditorGUILayout.LabelField(label or property.displayName)
        elif showBacking == "Always":
            NodeEditorGUILayout.DrawPropertyField(property, label, includeChildren)

    @staticmethod
    def LabelField(label: Any, *options: Any) -> None:
        # Implement a method to draw a label field similar to EditorGUILayout.LabelField in Unity
        # Placeholder implementation for example purposes
        print(f"Drawing label field: {label}")

    @staticmethod
    def PortField(port: Any, *options: Any) -> None:
        NodeEditorGUILayout.PortFieldWithLabel(None, port, *options)

    @staticmethod
    def PortFieldWithLabel(label: Any, port: Any, *options: Any) -> None:
        if port is None:
            return

        content = label or port.fieldName
        if port.direction == "Input":
            NodeEditorGUILayout.LabelField(content, *options)
            rect = Rectangle((0, 0), 16, 16)  # Placeholder for the position of the port handle
            plt.gca().add_patch(rect)
        elif port.direction == "Output":
            NodeEditorGUILayout.LabelField(content, *options)
            rect = Rectangle((100, 0), 16, 16)  # Placeholder for the position of the port handle
            plt.gca().add_patch(rect)

    @staticmethod
    def AddPortField(port: Any) -> None:
        if port is None:
            return
        rect = Rectangle((0, 0), 16, 16)  # Placeholder for the position of the port handle
        plt.gca().add_patch(rect)

    @staticmethod
    def PortPair(input_port: Any, output_port: Any) -> None:
        plt.gca().add_patch(Rectangle((0, 0), 16, 16))  # Placeholder for the input port position
        plt.gca().add_patch(Rectangle((100, 0), 16, 16))  # Placeholder for the output port position

    @staticmethod
    def DrawPortHandle(rect: Rectangle, backgroundColor: Any, typeColor: Any) -> None:
        plt.gca().add_patch(rect)

    @staticmethod
    def InstancePortList(fieldName: str, type_: Type, serializedObject: Any, io: str, connectionType: str = "Multiple", typeConstraint: str = "None", onCreation: Callable = None) -> None:
        node = serializedObject.targetObject
        instancePorts = [port for port in node.InstancePorts if port.fieldName.startswith(fieldName)]
        list_widget = ReorderableList(instancePorts, serializedObject, fieldName, type_, io, connectionType, typeConstraint)
        list_widget.display()

    @staticmethod
    def CreateReorderableList(fieldName: str, instancePorts: List[Any], arrayData: Any, type_: Type, serializedObject: Any, io: str, connectionType: str, typeConstraint: str, onCreation: Callable) -> Any:
        # Implement logic to create a reorderable list similar to Unity's ReorderableList
        list_widget = ReorderableList(instancePorts, serializedObject, fieldName, type_, io, connectionType, typeConstraint)
        return list_widget


class ReorderableList:
    def __init__(self, instancePorts: List[Any], serializedObject: Any, fieldName: str, type_: Type, io: str, connectionType: str, typeConstraint: str):
        self.instancePorts = instancePorts
        self.serializedObject = serializedObject
        self.fieldName = fieldName
        self.type_ = type_
        self.io = io
        self.connectionType = connectionType
        self.typeConstraint = typeConstraint

    def display(self) -> None:
        for index, port in enumerate(self.instancePorts):
            print(f"Port {index}: {port.fieldName}")
            NodeEditorGUILayout.PortField(port)

    def add_item(self) -> None:
        # Logic for adding an item to the list
        pass

    def remove_item(self, index: int) -> None:
        # Logic for removing an item from the list
        pass

    def reorder_item(self, old_index: int, new_index: int) -> None:
        # Logic for reordering items in the list
        pass


# Example usage
serialized_obj = {
    "targetObject": {
        "InstancePorts": []
    }
}
NodeEditorGUILayout.InstancePortList("exampleField", str, serialized_obj, "Input")
plt.show()

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict
from enum import Enum
import numpy as np


class NodeActivity(Enum):
    Idle = 0
    HoldNode = 1
    DragNode = 2
    HoldGrid = 3
    DragGrid = 4


class RerouteReference:
    def __init__(self, port=None, connection_index=0, point_index=0):
        self.port = port
        self.connection_index = connection_index
        self.point_index = point_index

    def insert_point(self, pos):
        self.port.get_reroute_points(self.connection_index).insert(self.point_index, pos)

    def set_point(self, pos):
        self.port.get_reroute_points(self.connection_index)[self.point_index] = pos

    def remove_point(self):
        self.port.get_reroute_points(self.connection_index).pop(self.point_index)

    def get_point(self):
        return self.port.get_reroute_points(self.connection_index)[self.point_index]


class NodeEditorWindow:
    current_activity = NodeActivity.Idle
    is_panning = False
    drag_offset = []

    def __init__(self, graph, zoom=1.0, pan_offset=None):
        self.graph = graph
        self.zoom = zoom
        self.pan_offset = pan_offset or np.array([0, 0])
        self.dragged_output = None
        self.dragged_output_target = None
        self.dragged_output_reroutes = []
        self.hovered_node = None
        self.hovered_port = None
        self.hovered_reroute = RerouteReference()
        self.selected_reroutes = []
        self.drag_box_start = np.array([0, 0])
        self.pre_box_selection = []
        self.pre_box_selection_reroute = []
        self.selection_box = Rectangle((0, 0), 0, 0, fill=False, edgecolor='r')
        self.is_double_click = False
        self.port_connection_points = {}
        self.node_sizes = {}

    def on_gui(self):
        fig, ax = plt.subplots()
        ax.add_patch(self.selection_box)
        plt.show()

    def controls(self, event):
        if event.name == "motion_notify_event":
            pass
        elif event.name == "scroll_event":
            old_zoom = self.zoom
            if event.step > 0:
                self.zoom += 0.1 * self.zoom
            else:
                self.zoom -= 0.1 * self.zoom
            if NodeEditorPreferences.get_settings().zoom_to_mouse:
                self.pan_offset += (1 - old_zoom / self.zoom) * (self.window_to_grid_position(event.x, event.y) + self.pan_offset)
        elif event.name == "button_press_event":
            if event.button == 1:
                self.dragged_output_reroutes.clear()

                if self.is_hovering_port:
                    if self.hovered_port.is_output:
                        self.dragged_output = self.hovered_port
                    else:
                        self.hovered_port.verify_connections()
                        if self.hovered_port.is_connected:
                            node = self.hovered_port.node
                            output = self.hovered_port.connection
                            output_connection_index = output.get_connection_index(self.hovered_port)
                            self.dragged_output_reroutes = output.get_reroute_points(output_connection_index)
                            self.hovered_port.disconnect(output)
                            self.dragged_output = output
                            self.dragged_output_target = self.hovered_port
                elif self.is_hovering_node and self.is_hovering_title(self.hovered_node):
                    self.current_activity = NodeActivity.HoldNode
                    self.pre_box_selection = [self.hovered_node]
                    self.is_double_click = (event.dblclick)
                elif not self.is_hovering_node:
                    self.current_activity = NodeActivity.HoldGrid
                    self.pre_box_selection = []
            elif event.button == 3:
                self.is_panning = True
        elif event.name == "button_release_event":
            self.is_panning = False
            if self.current_activity == NodeActivity.HoldNode and self.is_double_click:
                self.center_node(self.hovered_node)
            self.current_activity = NodeActivity.Idle
        elif event.name == "motion_notify_event" and self.is_panning:
            self.pan_offset += np.array([event.x, event.y]) * self.zoom

    @property
    def is_dragging_port(self):
        return self.dragged_output is not None

    @property
    def is_hovering_port(self):
        return self.hovered_port is not None

    @property
    def is_hovering_node(self):
        return self.hovered_node is not None

    @property
    def is_hovering_reroute(self):
        return self.hovered_reroute.port is not None

    def window_to_grid_position(self, x, y):
        return np.array([x, y]) * self.zoom + self.pan_offset

    def grid_to_window_position(self, pos):
        return (pos - self.pan_offset) / self.zoom

    def draw_connection(self, start_point, end_point, color):
        plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color=color)

    def draw_port_handle(self, rect, background_color, type_color):
        plt.gca().add_patch(Rectangle(rect[:2], rect[2], rect[3], fill=True, edgecolor=type_color, facecolor=background_color))

    def is_hovering_title(self, node):
        mouse_pos = plt.ginput(1)
        node_pos = self.grid_to_window_position(node.position)
        width = self.node_sizes.get(node, [200, 0])[0]
        window_rect = Rectangle(node_pos, width / self.zoom, 30 / self.zoom)
        return window_rect.contains_point(mouse_pos[0])

    def center_node(self, node):
        node_dimension = self.node_sizes.get(node, np.array([0, 0])) / 2
        self.pan_offset = -node.position - node_dimension

    def recalculate_drag_offsets(self, event):
        self.drag_offset = [np.array([0, 0])] * (len(self.pre_box_selection) + len(self.selected_reroutes))
        for i, obj in enumerate(self.pre_box_selection):
            if isinstance(obj, Node):
                self.drag_offset[i] = obj.position - self.window_to_grid_position(event.x, event.y)
        for i, reroute in enumerate(self.selected_reroutes):
            self.drag_offset[len(self.pre_box_selection) + i] = reroute.get_point() - self.window_to_grid_position(event.x, event.y)

    def draw_dragged_connection(self):
        if self.is_dragging_port:
            col = NodeEditorPreferences.get_type_color(self.dragged_output.value_type)
            from_rect = self.port_connection_points.get(self.dragged_output)
            from_pos = from_rect.center
            col[3] = 1.0 if self.dragged_output_target else 0.6
            to_pos = np.array([0, 0])
            for point in self.dragged_output_reroutes:
                to_pos = point
                self.draw_connection(from_pos, to_pos, col)
                from_pos = to_pos
            to_pos = self.port_connection_points.get(self.dragged_output_target).center if self.dragged_output_target else self.window_to_grid_position(plt.ginput(1))
            self.draw_connection(from_pos, to_pos, col)
            bg_col = [0, 0, 0, 0.6]
            fr_col = col
            for point in self.dragged_output_reroutes:
                rect = [point[0] - 8, point[1] - 8, 16, 16]
                self.draw_port_handle(rect, bg_col, fr_col)

import numpy as np

class MeshGenerationBurst:
    def __init__(self, planet, quad):
        self.planet = planet
        self.quad = quad
        self.is_completed = False
        self.vertices = None
        self.normals = None
        self.mesh_offset = None
        self.uv = None
        self.colors = None
        self.uv4 = None
        self.job = None

    def start_generation(self):
        length = len(self.planet.quad_arrays.extended_plane)
        self.vertices = np.array(self.planet.quad_arrays.extended_plane)
        self.normals = np.zeros((length, 3))
        self.mesh_offset = np.zeros((1, 3))
        self.uv = np.zeros((self.planet.quad_size * self.planet.quad_size, 2))
        self.colors = np.zeros((length, 4))
        self.uv4 = np.zeros((length, 2))

        level_constant, offset_x, offset_y = MeshGeneration.calculate_uv_constants(self.quad)
        
        self.job = MeshGenerationJob(
            vertices=self.vertices,
            normals=self.normals,
            mesh_offset=self.mesh_offset,
            uv=self.uv,
            colors=self.colors,
            uv4=self.uv4,
            triangles=self.planet.quad_arrays.tris_native,
            side_length=self.planet.quad_size,
            scale=self.quad.scale,
            rotation=self.quad.rotation,
            tr_position=self.quad.tr_position,
            radius=self.planet.radius,
            height_inv=self.planet.height_inv,
            level_constant=level_constant,
            offset_x=offset_x,
            offset_y=offset_y,
            uv_scale=self.planet.uv_scale
        )
        self.job.execute()
        self.is_completed = True

    def apply_to_mesh(self, mesh):
        if not self.is_completed:
            return
        
        length = self.planet.quad_size * self.planet.quad_size
        vertices_v = self.vertices[:length]
        normals_v = self.normals[:length]
        uv_v = self.uv[:length]
        uv4_v = self.uv4[:length]
        colors_v = self.colors[:length]

        self.quad.mesh_offset = self.mesh_offset[0]
        mesh.vertices = vertices_v
        mesh.normals = normals_v
        mesh.uv = uv_v
        mesh.uv4 = uv4_v
        mesh.colors = colors_v

    def dispose(self):
        self.vertices = None
        self.normals = None
        self.mesh_offset = None
        self.uv = None
        self.colors = None
        self.uv4 = None

class MeshGenerationJob:
    def __init__(self, vertices, normals, mesh_offset, uv, colors, uv4, triangles, side_length, scale, rotation, tr_position, radius, height_inv, level_constant, offset_x, offset_y, uv_scale):
        self.vertices = vertices
        self.normals = normals
        self.mesh_offset = mesh_offset
        self.uv = uv
        self.colors = colors
        self.uv4 = uv4
        self.triangles = triangles
        self.side_length = side_length
        self.scale = scale
        self.rotation = rotation
        self.tr_position = tr_position
        self.radius = radius
        self.height_inv = height_inv
        self.level_constant = level_constant
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.uv_scale = uv_scale

    def execute(self):
        side_length_1 = self.side_length - 1
        num_vertices = self.side_length * self.side_length

        height = 0.0
        normalized = np.zeros(3)

        for i in range(num_vertices):
            self.vertices[i] = self.get_position(self.vertices[i], self.scale, self.rotation, self.tr_position, self.radius, self.height_inv, height, normalized)
            
            if i == 0:
                self.mesh_offset[0] = self.vertices[0]
                down = self.normalize(-self.mesh_offset[0])

            self.vertices[i] -= self.mesh_offset[0]

            self.calculate_uv_cube(i, side_length_1)
            self.calculate_uv_quad(i, side_length_1)

            self.colors[i] = [0, 1, 0, 0]
            self.uv4[i] = [0, 0]

        for i in range(num_vertices, len(self.vertices)):
            self.vertices[i] = self.get_position(self.vertices[i], self.scale, self.rotation, self.tr_position, self.radius, self.height_inv)
            self.vertices[i] -= self.mesh_offset[0]

        self.calculate_normals(num_vertices)

    def calculate_normals(self, num_vertices):
        for i in range(0, len(self.triangles), 3):
            p1 = self.vertices[self.triangles[i]]
            p2 = self.vertices[self.triangles[i + 1]]
            p3 = self.vertices[self.triangles[i + 2]]

            l1 = p2 - p1
            l2 = p3 - p1

            normal = np.cross(l1, l2)

            n = self.triangles[i]
            if n < num_vertices:
                self.normals[n] += normal

            n = self.triangles[i + 1]
            if n < num_vertices:
                self.normals[n] += normal

            n = self.triangles[i + 2]
            if n < num_vertices:
                self.normals[n] += normal

        for i in range(num_vertices):
            self.normals[i] = self.normalize(self.normals[i])

    @staticmethod
    def get_position(vertex, scale, rotation, tr_position, radius, height_inv, height=0, normalized=np.zeros(3)):
        vertex *= scale
        vertex = np.dot(rotation, vertex)
        vertex += tr_position
        vertex = MeshGenerationJob.normalize(vertex)
        normalized = vertex
        height = MeshGenerationJob.height_at_xyz(vertex)
        vertex *= radius
        vertex -= tr_position
        vertex *= (height_inv + height) / height_inv
        return vertex

    @staticmethod
    def height_at_xyz(pos):
        lacunarity = 2.0
        gain = 0.5
        sum_ = 0.0
        ampl = 1.0

        for i in range(20):
            sum_ += MeshGenerationJob.snoise(pos) * ampl
            pos *= lacunarity
            ampl *= gain

        return sum_

    @staticmethod
    def normalize(x):
        norm = np.linalg.norm(x)
        return x / norm if norm != 0 else x

    @staticmethod
    def snoise(v):
        # Simplex noise implementation
        raise NotImplementedError("Simplex noise function is not implemented")

    def calculate_uv_cube(self, i, side_length_1):
        x = (i // self.side_length) / side_length_1
        y = (i % self.side_length) / side_length_1
        scale = self.uv_scale / self.level_constant
        x *= scale
        y *= scale
        x += self.offset_x
        y += self.offset_y
        y *= -1
        self.uv[i] = [x, y]

    def calculate_uv_quad(self, i, side_length_1):
        x = (i // self.side_length) / side_length_1
        y = (i % self.side_length) / side_length_1
        x *= self.level_constant
        y *= self.level_constant
        y = -y
        self.uv[i] = [x, y]
