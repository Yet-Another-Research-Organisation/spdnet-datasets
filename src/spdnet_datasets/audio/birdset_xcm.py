"""BirdSet XCM: bird species from focal Xeno-Canto recordings (Rauch et al., 2024).

XCM is the "medium" training set of BirdSet: focal Xeno-Canto recordings
(Vorbis, 32 kHz, variable length, median 32 s) of the species of the 8
soundscape test sets.

.. warning::
   **Partial local copy.** Only 21 parquet shards out of at least 341 were downloaded
   (``xcm_00000``, ``xcm_00017``, ..., ``xcm_00340``), i.e. 5 371 recordings;
   the shards are sorted by species, so only 136 of the 411 species are
   present. The official BirdSet test sets (soundscapes) are absent.

Samples: species with at least ``min_recordings`` (10) recordings in the copy
(95 species, 5 185 recordings). ``ebird_code_multilabel`` holds one species
per row, so the task is single-label. The atlas measured 10 pairs of identical
5 s excerpts under two Xeno-Canto identifiers; they are kept.

Covariances
-----------
One covariance per recording over its non-silent frames (whole recording by
default; ``crop_s`` restricts it to an excerpt of ``crop_s`` seconds centred on
the first detected event, as the atlas sheet did with 5 s): ``p = 20`` MFCC
(default) or 64 log-mel bands.

Splits
------
``"group_kfold"`` (= ``"recommended"``): 5-fold ``StratifiedGroupKFold`` on the
species grouped by recordist (seed 0); test = fold ``fold``, validation = the
next fold, train = the others. One recordist often records the same individual
or site, hence the grouping. Empty recordist names form one group ``"NA"``.

Download
--------
Access: direct (public Hugging Face repository ``DBD-research-group/BirdSet``,
no account), branch ``data`` pinned at commit :data:`HF_COMMIT`, folder ``XCM/``:
``XCM_metadata.parquet`` (16.7 MB, 89 795 recordings) and 182 tar.gz shards
``XCM_shard_0001`` ... ``XCM_shard_0182`` of Vorbis files (324 to 609 MB each,
95.9 GB in total). The sha256 of every file (Hugging Face LFS identifiers) are
registered (:data:`FILES`, :data:`SHARD_SHA256`). Licence: CC BY-NC 4.0 (dataset
card), each recording under its own Creative Commons licence (``license`` column
of the metadata).

The local ``xcm_<k>.parquet`` shards are a partial re-export that matches no
published file. The published data are read directly instead: the metadata gives
the label (``ebird_code``), recordist, length and detected events of each
recording, joined with the tar members through ``filepath``. Resolution order:

1. ``data_dir`` (default ``<data root>/AUDIO/BirdSet_XCM``) exists: the local
   ``xcm_*.parquet`` copy is read;
2. ``archive=`` given: a folder holding ``XCM_metadata.parquet`` and
   ``XCM_shard_<k>.tar.gz`` files (e.g. ``huggingface-cli download
   DBD-research-group/BirdSet --repo-type dataset --revision data --include
   "XCM/*"``); the files are checked and the shards extracted into the cache
   (the user's files are kept);
3. otherwise ``shards=`` must be given: the metadata and the chosen shards are
   downloaded into ``<cache_dir>/birdset_xcm/downloads/``, checked, extracted and
   the archives deleted. ``shards=k`` takes ``k`` shards evenly spread over the
   182 (they are sorted by species: the first ones hold only a few species),
   ``shards=[1, 50]`` lists them, ``shards="all"`` takes the whole 95.9 GB.

Example: ``fetch_birdset_xcm(data_dir="absent", shards=4)`` (about 2 GB), or
``fetch_birdset_xcm(archive="~/hf/BirdSet/XCM")``.
"""

import json
import warnings
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np

from ..core import check_local, dataset_path, download, run_pipeline
from ..core.fetch import sha256sum
from ..manager import DatasetManager
from ._common import (
    SequenceCovarianceDataset,
    downloads_dir,
    estimate_stage,
    evenly_spaced,
    folds_to_split,
    kfold_assignment,
    load_result,
)
from ._features import iter_parquet_rows, process_recordings

DOMAIN, NAME = "AUDIO", "BirdSet_XCM"
HF_REPO = "DBD-research-group/BirdSet"
HF_COMMIT = "806ed2cda4ddcbe6efa194ccafff930aa0e557ce"  # branch "data", 2026-11-15
_HF = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_COMMIT}/XCM/"
METADATA = "XCM_metadata.parquet"
FILES = {  # url -> sha256 (Hugging Face LFS oid)
    _HF + METADATA: "62a3aac98bf48e1ca05cbe3e71cf367c966dcdbd6dfc809ad1f9e7a6b9004bbd",
}
SHARD_SHA256 = (  # sha256 (Hugging Face LFS oid) of XCM_shard_0001.tar.gz ... 0182
    "17789615ba27548705bd6a50090bcd6cc3c1155876ecd274305a2fc3fc288c7f",  # 0001
    "df2c2bda5e5c96e5bf0c1949327d408ccdbd31028202c0c40f9b93e6df40fd7d",  # 0002
    "91419282a9ab9aec419aac620cf46fa510df3e6f6514d4b34a50610f2d406e3f",  # 0003
    "ebc6f789f05f70f59f41968dda0a9602c47c1c8cc98f458f5bd2f80fcce5fa95",  # 0004
    "f04cdcde3b3317bfb94d54b5d42aea11c2e5db343be5ca3aeb9c8f32c190b6e0",  # 0005
    "193fb38870943d84abb16653fe4cff49e4324a68b19854aaf5eba8112e55d1d2",  # 0006
    "3442314780dc76a393a7e6e85df271738e288fc6fbb7077f85c494012f8a295d",  # 0007
    "f51c62f08d046a4ad0288cec7746ca57df9a11d53a59137af345f679bdd2d63e",  # 0008
    "198bf45222ca39b82aae74f2d63c76210bf438cebba2b4c22d484d688403c0e8",  # 0009
    "5d2fd9393f1d6609a40edacd2dcff7e4588d1af47317295a5f4173c8a3550141",  # 0010
    "0f651053f60fc49a7faff5c85efc18dedd51d8a7d38bd70b49a81e26d1369969",  # 0011
    "f4788b52404871046dc3524420d5e4db60a2cbb45135910cac5bbc867002281b",  # 0012
    "6df4cba1feeda65306649cd46480bcf95a25296d899a7610fa614f9c2cda43e9",  # 0013
    "8dfb2129ebe1be038308fd42af2b3fa0e34b8fcaa74ca32bfb063f9590c5de3f",  # 0014
    "0fb98dbb9ef5a302e29b3ffa83b42618590b2c374e82f5caff609cf79a29c2be",  # 0015
    "d3cd649377eba30877515360386f185279966183e5dc0074d1ad4094bde61d80",  # 0016
    "8416cdb97e0f6ec79a09ad1995dc0fc4b7f3c473065aa12dab2d04368c3f6519",  # 0017
    "2227a0e3f4ea41f809944939e193b2e4dba74eac447f1b3e0eb0580edcef41ac",  # 0018
    "092556105f4a72d16da0be5851d9eca626a91d18e2c4492346f27c6fb90134c9",  # 0019
    "8072c677c41a567e57e99931c7bafb7b344ba834f8e66fb5ff0e61f09973fdd6",  # 0020
    "2ede6fd24697c62b4045df7739964f09f849adbb9bba7938af8931b133c43d9b",  # 0021
    "08373c6ddef23fee528c2e41c3234262dd44cf60ef648f36ac657c29aff8c403",  # 0022
    "30ba8ee25f9f8ea6af217c65c0297f3b17cd0d53c30d8cc79d4d9b19bbe17540",  # 0023
    "cb2d2c00e0cda93a8b656389505be931ca6ff7430c54a42a62a404a848ad0c9d",  # 0024
    "7e5ee822cc9b964f639b46759b3a4a584846307d888369ada389161aa1768856",  # 0025
    "020cdec91c8f3699c4d801892e1dbb0506bfb9993832b70af738661244ae4554",  # 0026
    "39e4512f44bcac38049d5c8376aa885f41c906f4361b3fac745e7e009d8b98d3",  # 0027
    "1a0e598a5e9097012cd2f34091341bf63e53889d09499603c66bf3776a56f299",  # 0028
    "bdf68301cd860d019090b704ca33218b9f7b116c6b3f0b56a42d15a68ba72d35",  # 0029
    "8c50e794b71e5876c8f8c089540c5c586900c5ff4b58ce1ce716fe3f7bd0c280",  # 0030
    "7e9a5dd82409c56986ddc243f1b050838538e9f042f5b36ed0b91a1a17ca6414",  # 0031
    "58f6422a85dda880b1279228f9256805731fd39dfcd1a2df4588ac343e2df71f",  # 0032
    "45fbeaf55435e1606c258cc66ac9207e7269b5a2a13a9a29d00b4502b015a543",  # 0033
    "30830c81594aba1342f81ece777ed13add93750f41cd2c4aca8d841f3aa65228",  # 0034
    "2fcb0ea3894822fdf98f55ea5528be4209a83e8d7b4ef42a47c46710dae3bf84",  # 0035
    "7c3ad1cac52327749662e58bccce9086caed7955edf4fea455c7f306849de3c7",  # 0036
    "07b342ca241993b6c81735bae96251cfba893beb7cce33fc79f91dcc97b957ec",  # 0037
    "891c273a6f1e88c4beb49da8e799709ae50512585fd407caa0a7bb0decf8d701",  # 0038
    "4a2a026944011bbb24adc0e5f90c274e2df3f8c7fd85009e5dc12087c08a308e",  # 0039
    "7dc614d291f505185ea9845f23264d337fedd0176e51f8dfd7fadabdaad3dad2",  # 0040
    "f440c7e4ea17caeef39cd98cb6fa2726d1cee0bed54dc46c8f5334f84c2bd4c4",  # 0041
    "96ba3bd20ffa2a783c3854017e68835022a85090f1fb696f7f3a658f64f1d5b5",  # 0042
    "054ccd5f65867ac1da0542d0918c196a5d0799c48cb63341afeb2812719a9285",  # 0043
    "4971dac6ec4ad8a4d0ca4076c3d056bcd837a17fbf0383bf682ea6f1d336abe5",  # 0044
    "81e893164b1df55320b526f219c4832aa7bbf7bc07fb2127b40f3fd9c4ead0fa",  # 0045
    "77e578f5e53ef540a06b0ce5154e4359fe1706092c8f44f425dedccb783fa92f",  # 0046
    "3d094472949924b157d33240c832309e28dcf501381b12b32e4215c54019358f",  # 0047
    "b6dc1d484a43f4a8abc71e37482c3f11faf0525436ff9dcab22bc36e5b3bc06c",  # 0048
    "5233944ee0df58c2bd98d9fe1cf577ea333b44af949df15eaf39d8cb0127429a",  # 0049
    "52376f6d43d9ed7e0787f69f3ffdf74bec410d3081c98431cd0b36a3bb791a90",  # 0050
    "bcd7d96aa2a42c7865e7f6f94e3b734adfc5760053cf92cca4929e5b3db888b1",  # 0051
    "87bd2b25a2627c6de0be6afa6aa396d3ad8d957816d6f8ee428719251ca686e3",  # 0052
    "cc3d6d8ef71686a522f9e5f1ee1a0a96964718a9ecdaccdfa4a890def57ac7dd",  # 0053
    "83d6f984a07990ac78d66327457ab73f7f524971d8fc6269e48046f6484ffe99",  # 0054
    "54f1ebee868ee08eaa9223a244a48621f9e79d81863ddbad91fe7c72ee4820e2",  # 0055
    "9f9f394ae0db050333584697c3156fe9ce791245e7dc9fefcac96700bad9c347",  # 0056
    "f73c161ec432fbbb053d3ce29bbacfbbdca2c85dcbb1c152c4c69663339317c2",  # 0057
    "93c34f03a2c4953ec4e4a1850b24b96a5ac28311e85a876bc2c7c1228606c11a",  # 0058
    "85c3d646bc7d40c69ad5cae771d51a7fd81de52abd734f0db93fb467b4198423",  # 0059
    "5ac7c8e9c68142d54634385f12512105a403186e89d33d03737bb3ce9d27f7ff",  # 0060
    "ec9b6c7717f84227894d6b0b9c19e2a75d20ab9dba427e7badce959b71e09077",  # 0061
    "cf857e663cf970bb8d6ef448a73506da08520054b69ca02c5279a4152253083a",  # 0062
    "33358017fbd6c1dd8fa357cb2fcc5c990319a39fb366d5ff110d56adcabf37d9",  # 0063
    "aad51dbae118e95f42b24725bf659c01c00c7658bca1fd0e17ff4cee528c4a0f",  # 0064
    "9c4b95b1f8fdc3231e2c33c0a07b832ee4aaec90a313259268d04ed6a209f7a0",  # 0065
    "66d6452924f7982fdc7e340170fe8ba946cee446ed0abcfa6092b9d31ca5176c",  # 0066
    "3eb2ab8be2d12dd6c3ba6b50dbe16deedd3d7a5dd3856cbf5105c21b39413412",  # 0067
    "e4a0f15f60fc607771956b8052b52f88462c298ced26c8c30831fa34c103f02b",  # 0068
    "04c92dea4cab8de5cd3c7de069ffb7f0187d06df94720b8b745e51ccd0827e43",  # 0069
    "82176347dc8192f20f94a7881c82710a6dc214a6ac136ce358d6a7ccc64ed0a1",  # 0070
    "8c119dc22d8d36aa35ff9fec0aa3148b8883372414a9fb0a7be304539416bd11",  # 0071
    "dbfbce69e4255f8f459be3b1b51ce00b73e1819bf48ae0bc479cce75c9730d2c",  # 0072
    "231485603977674ad66eb59a14c8160052556d8642c9df2b387130e94f286b48",  # 0073
    "d01f9a1d52ab40fbabb0ccbab373ae04cf78aa0fda0d1ae173749c35c32b4123",  # 0074
    "69fb9daf8d58907ddd724327f7056f0584c90767dd70129a4e3c421c87b26ecc",  # 0075
    "dd55c6fe76298a233c950243942909e0e149f3e5944140616a1dbf02c1855bd1",  # 0076
    "a22362348426748e6d1b611d5788407fe047b19b92c89a41e3fcbc83c5389816",  # 0077
    "79fd485dd59500281570606a91de6b1654c178455e6e4c3e4495eb0b7d984543",  # 0078
    "3aeedd113821c0ea79b2bc43c612a7b45239a5a182c9a92cec2863a0c6a34a88",  # 0079
    "0572e5508714eb2e8c35be06a74d3bfff858c551eb515c2ac9b919843ea8277c",  # 0080
    "7a11239ab1c0743e75ba6a8f4ab2c41a9f014dbcaf2b8288f4b12efbd0c262e7",  # 0081
    "bc85b9f1651382b5cf1465f71a7b9ede43f5ff82806f376561f2b317305bd2b5",  # 0082
    "e7a0cbceb6ff14209e3828b266e8f72fd80a10ab79e77420bb0d560468aa5bbd",  # 0083
    "37293bff9bb97beaaa4959285c2cf71948418a9273db93cc0ce1e19bfc708c0b",  # 0084
    "b88e767c7b1556c17b82113bfcaf4c100c1d26db397136feae1776988d7dfec7",  # 0085
    "16dba5b93c92fe9a2554948ed7bde1bf2c2bedddd8d8756e5fe3ca14a2bebfa9",  # 0086
    "9b77e2f19db0490d517bdef21945f459df31cae9a53eb4ebeb5cb23261a41f3a",  # 0087
    "8a0f094f3ce2305b567bb736dd60e01c9a3531f62ce5f696cb623f53ca1bb45a",  # 0088
    "a413eb5ac697c8c2040c17644e8a7f3343564d049447b8fe025ef8677d487111",  # 0089
    "79317f22ed2ddd15416f0afbdc391a17882f93240ae0f3ddbe5483323f568b31",  # 0090
    "b560639bc6b49c7cbdafa1fb810e304cccda36bf4e051f74a8dc7a58f248e85d",  # 0091
    "2caaacc2a6b57b6dda02338802fbc320b4b6aae1d2cdf411f4a73a56f1721e27",  # 0092
    "5d016df105b66de1ac6fdcc35ed8359f9730bba91a1aebfa1c54e03982533b82",  # 0093
    "cd3335576e8d816eadc1020d765eebb8e08863fa627c971b023b24ad5ee8b2c9",  # 0094
    "8d73c4050833ddd0efb5777a80aedf60700990bae56d7c4ef688f2a41b2a3344",  # 0095
    "23ccb459be375caefcc2612cafe140ed11b158ba280f2f0eb24c249639591ead",  # 0096
    "25c236ea29f122e6b5e914909241aacc47f1f26f45cbc292287f8843d0016529",  # 0097
    "74927bed76b15d871f510ab399f51fc30262783d74a27caeb6830fbe82613892",  # 0098
    "fae820d4c3fa1a328652de22a5bfd531a467e023943df53aabf34f534f206569",  # 0099
    "b112e85c93d9b3840a672ad265b07080837d43e25f72eabf51187dd72f4ea498",  # 0100
    "398ca8fcfa56c279a655e128cf929af2eb3aec08fc205d97150f075f4d7a3049",  # 0101
    "73c08e4f5221a7517870e50791bbe55cb549d02287ecbfc806c53f1f8f776c86",  # 0102
    "dfc1931aa97e487c34b9b31d931acd5e362214fc3fb45dfe433170c17f4eb470",  # 0103
    "409af5549c384e0335c204c1fddadcf95ab47d3f55a8b5a7f8e6b031f38ed991",  # 0104
    "0a7a2d831c48ebb6957ad51ec52e9b3f82f916567f8c8da2047f78e26c26adca",  # 0105
    "1510d93c984ba92931d5f83229d84cef1650709265b3532feffb864e8b0ef853",  # 0106
    "b3912a4a48411a9cdab1906dfb03f44ffacced1371a9196fb4f52bede446b54a",  # 0107
    "99df695dace6b2869c2edc24fc09aabfa020b48ea08ea12553b42cdc880cdaa5",  # 0108
    "662e5901654de4be5c979b272b6dfee99e02fa336c58ac294c98bd4fbcb17bff",  # 0109
    "b1dd3fdf18b64cca999fbab22b59290af8949618e121ad2c6aadc3ad68bb780c",  # 0110
    "dfc272b400adcfaebab0693df702b0768b133c0bc9105075a91390f79255f3b3",  # 0111
    "ac67180cab1133c715c42f68468ea909272c010d683d86c6a160c7dd096a6df8",  # 0112
    "28e3cdb8390ab2718dd10fb76b21c75595e4d0538bb40f1fb9f444f46ba6ec0c",  # 0113
    "a2ba0846af0170094c9a65829c5b37638952bf31c3b7dd635f2bb23a73f6833d",  # 0114
    "05012e7e997cb19f2cf4f4cbffec6c80638a9cb7c71b35637b26c672ba0ec3d7",  # 0115
    "bd35a5ea395b6592b3e5953989f1e50f5db640ab7764c6dc22ae1f3f978d129d",  # 0116
    "c6be16ce5873466489fef333330bb468a30d305b5f747920239b10ac6b02dbd5",  # 0117
    "2f40aa48b29d496cf158b48a11e0bf4d38d781e3d75db31c1d54b5542766c5ff",  # 0118
    "cd72aff21631deb5e087ac7f0a9a96c6c3f737572a8ea6a1e42622a1eb2f8a41",  # 0119
    "4f03e2f7a82bd7f3feeecb91045ae8eb343a793a83e18bcc073d04f1d2a0c39e",  # 0120
    "28cc5207fb85ba14fc8e8cc52b26405359bf7b9f52e18e9e7a9a53e32010d1d9",  # 0121
    "e0142b200cf758a29e613a9cda1aa1d11a2f19b828b18267ea917fa55b473996",  # 0122
    "53e720b84e6d3ab9bdf37d12b9c2cf2600404a1dfc57a9ee8d7385e10258e7d8",  # 0123
    "163c6120f185bb68eb5e313e2a10e0bb16827ee054ad112f159d322812f0a645",  # 0124
    "31d4a73df1ead034d3aa0d67c6b37f1dd650e8764d48fc73748a15057fc2a80a",  # 0125
    "8c8ad48f7fe2f901ac8934871325cc940d4b88ab56822e76ef8df67b860139fa",  # 0126
    "0670afd79a7ae3db3196beacc9a49819906ceb2b10e3bc5bb67c747f02b2d4ee",  # 0127
    "66dd50bfbc8d74cd5019c45be3709cc284223f7c9d5deae3e1abb53b7b8552f0",  # 0128
    "18f9c0388d04eafde48a64bc0603083c24098bbd781aa6327965c3c669d0cff9",  # 0129
    "1ce4c05ca035332665fbb1ac0fe86ca1cb14de69e3a8066e44608c308d675f50",  # 0130
    "9d64fa2d815d61388c6d8b77d5b8ee3924392ae945361fbda6b1324ff80bf92d",  # 0131
    "6592106054d1e14982300d26838b8c06dba3bcedd523146165e63c6b8cc50a76",  # 0132
    "49c480626edc42a8613a906fae2d777e5caee266e70d2790a6d5a7f02d9a870c",  # 0133
    "da2c1211de55ae4e9c55840f3d5522a8a5d46717a3040f74d12eba2066e3ed07",  # 0134
    "47cbd85e907b88f151794c2d12774c5ade022da1de9b65c5ba848a181199593e",  # 0135
    "f88ce7362cc631f7c1cc0eb423d64f2cd2d117ecb88579d8406e768e914e6685",  # 0136
    "b9183d33908f6e1899b61b65fd1b20b12e5fdb08b48dca9c5a7c30bfeb2a1798",  # 0137
    "e9580bd8eaa10493c0dab15e9f5ff1e503d35c7d67983bde33f24beba0b89df2",  # 0138
    "e30223122fe4efe2ae77852c18205e1e22f7e328391c58d954f7ba822d4dc338",  # 0139
    "1322ff7261e44392a0d05ed573f20f7126db29ba006b04e73af9d935b5c33409",  # 0140
    "a8e4fa03e3f80ccee32eb453a4248ca8915954261c7a46b3ff1062723646634c",  # 0141
    "7a20b3364af1ec3ac9a7eeb7e1d0d20e68b9692707b58e4ef1538cff1058182c",  # 0142
    "bfd0ef87324aa643079569d2c9c7aef47873548ed25f40ecda761239d35b760d",  # 0143
    "6387a042fad9ada42a5c5b50aa0adcc5bc6ee2ff4be9659599ad5b8308c5aac7",  # 0144
    "30d7351c54eecb95eae94ced592edd938b17663b0082ac7c3c9ebd5325466bb9",  # 0145
    "c344aacce95d542698c6cd6fc7c08bed291644cf4320b290606ef4175878865b",  # 0146
    "c0b7f9d497ac7737f9cbb43426f5ec0cdc9b951450fa8fcf0325785bdb961d59",  # 0147
    "694395451cfe8354ff65aef0561d6d029a4601ecb506f4add27b0a39e6f07bb3",  # 0148
    "bfe4480837e881ec43731c4bac6e264924285e674e7accc7c3c3467f10116946",  # 0149
    "9ed0b3677bcc8a588ac3d57cbab94e59e01eda136498a60adc9b0f5e02b31caf",  # 0150
    "227bc3505970acae7988583d38c9158131d26c6e94ebdd0a8e89a302ec77ad52",  # 0151
    "b7f3efeb4a2be4963b43fc7d9cf6e9827a6527637fa005d677142f20ae3f2cf9",  # 0152
    "dc54a809196ae359134c0352b7fc0a70810d415ad580a2fda501e1f05cf8b573",  # 0153
    "3e9c4d7eb78d44e63805c48ad7efaecc170c726b6d173042e83d0dc3628837e0",  # 0154
    "1894f065d1de948fe94a2259f0a5ad720faba61fde2e0cc49d527dd00b7e06a9",  # 0155
    "2b995a6c74810b939557aa0ae7a346190b8b8a0eebd3fb1ed78c50d787aef560",  # 0156
    "fc46ecddc154584cf0f974c74e626c58919d6c18da3e8d7aafe2b6097ce08201",  # 0157
    "82faea0ad246b44b344d10a41dfc666b7cffcb7314aae579b8a8d946ba164735",  # 0158
    "86a8679775b8f418169e31b78a5b79eadef0130a7a0302443b20c025ebb397e9",  # 0159
    "4816793dd7ad1578495e55ff92a8c6ebc90f4389011b1cc3cf0ed6b0f359c219",  # 0160
    "38d9c75cd21e5fdc0884322d9f85f7479a033650e19ea1358a3daba730494232",  # 0161
    "c4df9fd7d5aab4829299236e821b5e91cfad7a74d33dc6ed93681b72c918723c",  # 0162
    "11880aef3d94eca072ae02df7505a975fd91d2c86d0b7e1da09e995766fb2981",  # 0163
    "acb3358abc9d7f7493c8d5922211c9e33b3fe130c064c45a6fb779e2e764a20d",  # 0164
    "3e9bf1cd05320edacd7675e1739b278617cad2f36f7ad154896656246b84cdab",  # 0165
    "29f0742ce08b10d429fd5f8a0155ff4d9d377aa0db026dc3c7ac4f825d27ac3d",  # 0166
    "b8f9f86eb475d8687394904127f5f785d008cce43f8771cb94fbe040897d92f1",  # 0167
    "e9099f04953c70f2da4a3e6eda4fe66c021dfbf7c1b9c56c610f9f69a393f12a",  # 0168
    "c159ba8843e3427523f21cc881e1c9ab7a526e67e6ccbc168cef1dff11403ec0",  # 0169
    "7705a3f4c393688dba68b2fe786296fb8f557fbca7fb0e92fa499648643cd17b",  # 0170
    "f0d053efb46c6281f4f8e7d3ec587b3e2806a9829e5c685bbf3d4eaecf4d210c",  # 0171
    "194b6aabbca607235cfb5b05882860b543e93a446c88f5d4ff30882bc50a8cc7",  # 0172
    "893d697b5131fddb781afd671d824f075d1f3611736686ee71ebca23de3b60ba",  # 0173
    "7ca8da91ecccaed7b3ecd4a5589477ef7d9d275278d9f527e70fd04d5cea735d",  # 0174
    "86e3571d76cb08ea50a26ea0fc3546e1b8a320d264cc11a542296991fcceb3a6",  # 0175
    "33a65d1df2d3206cd921edd435b652b5e5bda4c9b0e4cc2be7eddcd6b1015c06",  # 0176
    "5134880f371518ead7ac2d075918141c67c2d3973c19e1d5476eae998f1145a0",  # 0177
    "a9992c695c32776ad29c4ad77537e613612e7705cf90e2d20ee85c3651a44c46",  # 0178
    "9b5d7983d915e055dcfc5a0271317629cd06dae1ad4445a1ff78649afd3c41fe",  # 0179
    "664c52e0b12609bcb1dc5e1730fd3ca2dd19a0d3868211aa58f44ae66e1038e8",  # 0180
    "91a5ba5d1ef8ccbf2ddacb74475d9aadf45ff4b7839659fd08be5d58bb5156e8",  # 0181
    "09997eb7643aef77aa0575c126948092f2ae8e2e5e078bd103c109f9b1dfe688",  # 0182
)
N_SHARDS = len(SHARD_SHA256)
N_FOLDS = 5
PARTIAL_WARNING = ("BirdSet_XCM is a partial copy (21 shards, 136 of 411 species) without "
                   "the official soundscape test sets.")
INDEX_COLUMNS = ["filepath", "ebird_code", "recordist", "length", "detected_events"]


def shard_name(k):
    """``XCM_shard_<kkkk>`` (1-based)."""
    return f"XCM_shard_{k:04d}"


def select_shards(shards):
    """Sorted shard numbers (1..182) from ``"all"``, a count or a list (see module docstring)."""
    if isinstance(shards, str):
        if shards != "all":
            raise ValueError(f"shards must be 'all', an int or a list, got {shards!r}")
        return list(range(1, N_SHARDS + 1))
    if isinstance(shards, int):
        if not 1 <= shards <= N_SHARDS:
            raise ValueError(f"shards must be in 1..{N_SHARDS}, got {shards}")
        return sorted({int(k) for k in np.linspace(1, N_SHARDS, shards).round()})
    picked = sorted({int(k) for k in shards})
    if not picked or picked[0] < 1 or picked[-1] > N_SHARDS:
        raise ValueError(f"shard numbers must be in 1..{N_SHARDS}, got {picked}")
    return picked


def _extract_shard(tar_path, dest, k):
    """Check one shard archive against :data:`SHARD_SHA256` and extract it once."""
    from ..core.access import extract_archive

    if sha256sum(tar_path) != SHARD_SHA256[k - 1]:
        raise OSError(f"Checksum mismatch for {tar_path}")
    return extract_archive(tar_path, dest / shard_name(k))


def download_birdset_xcm(shards, cache_dir=None):
    """Download the metadata and ``shards`` into ``<cache_dir>/birdset_xcm/downloads/``.

    Each shard is checked, extracted once (``XCM_shard_<k>/``) and its archive
    deleted. Returns ``(metadata path, list of extracted shard folders)``.
    """
    dest = downloads_dir(cache_dir, "birdset_xcm")
    meta = download(_HF + METADATA, dest / METADATA, checksum=FILES[_HF + METADATA])
    dirs = []
    for k in select_shards(shards):
        out = dest / shard_name(k)
        if not (out / ".extracted").exists():
            tar = download(_HF + f"{shard_name(k)}.tar.gz", dest / f"{shard_name(k)}.tar.gz",
                           checksum=SHARD_SHA256[k - 1])
            _extract_shard(tar, dest, k)
            tar.unlink()
        dirs.append(out)
    return meta, dirs


def ingest_birdset_xcm(archive, shards=None, cache_dir=None):
    """Metadata and extracted shards from a folder of files downloaded from the Hub.

    ``archive`` holds ``XCM_metadata.parquet`` and ``XCM_shard_<k>.tar.gz``
    files; ``shards`` (as in :func:`select_shards`) restricts the shards used.
    The files are checked; the shards are extracted into the cache.
    """
    folder = Path(archive).expanduser()
    meta = folder / METADATA
    if not meta.is_file():
        raise FileNotFoundError(f"{meta} not found (download XCM/{METADATA} from {HF_REPO})")
    if sha256sum(meta) != FILES[_HF + METADATA]:
        raise OSError(f"Checksum mismatch for {meta}")
    present = sorted(int(p.name[10:14]) for p in folder.glob("XCM_shard_[0-9][0-9][0-9][0-9].tar.gz"))
    wanted = present if shards is None else [k for k in select_shards(shards) if k in present]
    if not wanted:
        raise FileNotFoundError(f"No XCM_shard_<k>.tar.gz file in {folder}")
    dest = downloads_dir(cache_dir, "birdset_xcm")
    return meta, [_extract_shard(folder / f"{shard_name(k)}.tar.gz", dest, k) for k in wanted]


def _files(root):
    files = sorted(root.glob("xcm_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No xcm_*.parquet file in {root}")
    return files


def _species_names(path):
    """eBird codes of the ``ebird_code`` ClassLabel (Hugging Face schema metadata)."""
    import pyarrow.parquet as pq

    meta = pq.ParquetFile(path).schema_arrow.metadata or {}
    try:
        info = json.loads(meta[b"huggingface"])["info"]["features"]["ebird_code"]
        return list(info["names"])
    except (KeyError, ValueError):
        return None


def _row(row, label):
    events = row["detected_events"] if row["detected_events"] is not None else []
    recordist = row["recordist"]  # None, "" or NaN (pandas) when missing
    return {"id": PurePosixPath(row["filepath"]).stem, "label": label,
            "group": recordist if isinstance(recordist, str) and recordist else "NA",
            "length": row["length"],
            "event": (events[0][0] + events[0][1]) / 2 if len(events) else np.nan}


def _kept(index, min_recordings):
    counts = index["label"].value_counts()
    return index["label"].isin(counts.index[counts >= min_recordings]).to_numpy()


def _index(root, min_recordings=10):
    """Metadata of every recording of the local copy and the mask of the kept species."""
    import pandas as pd

    files = _files(root)
    names = _species_names(files[0])
    index = pd.DataFrame([_row(row, names[row["ebird_code"]] if names else str(row["ebird_code"]))
                          for _, row in iter_parquet_rows(files, INDEX_COLUMNS)])
    return index, _kept(index, min_recordings)


def shard_index(metadata, shard_dirs, min_recordings=10):
    """Recordings of the extracted ``shard_dirs`` with their metadata (no audio decoded).

    ``filepath`` of ``XCM_metadata.parquet`` is the path of the audio file inside
    a shard. Returns the index (``file``: path of the audio file) and the mask of
    the kept species.
    """
    import pandas as pd

    files = {str(p.relative_to(d)): p for d in map(Path, shard_dirs) for p in d.rglob("*.ogg")}
    table = pd.read_parquet(metadata, columns=INDEX_COLUMNS)
    table = table[table["filepath"].isin(files.keys())]
    index = pd.DataFrame([{**_row(row, row["ebird_code"]), "file": str(files[row["filepath"]])}
                          for row in table.to_dict("records")],
                         columns=["id", "label", "group", "length", "event", "file"])
    return index, _kept(index, min_recordings)


def _process(source, out_dir, *, root, shards, limit, min_recordings, crop_s, features,
             n_mfcc, n_mels):
    if shards is None:  # local xcm_*.parquet copy
        index, kept = _index(root, min_recordings)
    else:
        index, kept = shard_index(*shards, min_recordings=min_recordings)
    candidates = np.flatnonzero(kept)
    keep = candidates[evenly_spaced(len(candidates), limit)]

    def blobs():
        if shards is None:
            for pos, row in iter_parquet_rows(_files(root), ["audio"], keep):
                yield pos, row["audio"]["bytes"]
        else:
            for pos in keep:
                yield pos, Path(index.file.iloc[pos]).read_bytes()

    def records():
        for pos, blob in blobs():
            meta = index.iloc[pos]
            rec = {"id": meta.id, "label": meta.label, "group": meta.group, "blob": blob}
            if crop_s is not None:
                centre = 0.0 if np.isnan(meta.event) else meta.event
                rec["crop"] = (max(0.0, centre - crop_s / 2), crop_s)
            yield rec

    process_recordings(records(), out_dir, features=features, n_mfcc=n_mfcc, n_mels=n_mels)


def birdset_xcm_splits(meta, labels, groups, fold=1, seed=0):
    """``StratifiedGroupKFold`` by recordist (see module docstring)."""
    parts = folds_to_split(kfold_assignment(labels, groups, N_FOLDS, seed), fold, N_FOLDS)
    return {"group_kfold": parts, "recommended": parts}


def fetch_birdset_xcm(data_dir=None, cache_dir=None, keep=None, estimator="scm", force=False,
                      verify="size", features="mfcc", n_mfcc=20, n_mels=64, limit=None,
                      fold=1, min_recordings=10, crop_s=None, shards=None, archive=None):
    """Covariances of the BirdSet XCM recordings (partial copy, see module docstring).

    Parameters are those of :func:`spdnet_datasets.audio.esc50.fetch_esc50`
    (``data_dir`` default ``<data root>/AUDIO/BirdSet_XCM``), plus:

    shards : int, list of int or "all", optional
        Published shards to download (or to take from ``archive``) when
        ``data_dir`` does not exist; required without ``archive``.
    archive : path, optional
        Folder with ``XCM_metadata.parquet`` and ``XCM_shard_<k>.tar.gz`` files.

    min_recordings : int
        Species with fewer recordings in the copy are left out.
    crop_s : float, optional
        Excerpt length (s) centred on the first detected event; ``None`` uses
        the whole recording.

    Returns
    -------
    Bunch
        ``covariances``, ``means``, ``labels`` (eBird codes), ``groups``
        (recordist), ``meta``, ``excluded``, ``splits``, ``processed_path``,
        ``covariance_path``.
    """
    root = dataset_path(DOMAIN, NAME, data_dir)
    published = None
    if root.is_dir():
        check_local(root, verify=verify)
        source = str(root.resolve())
    else:
        if archive is not None:
            published = ingest_birdset_xcm(archive, shards, cache_dir)
        elif shards is None:
            raise ValueError(
                f"No local BirdSet XCM copy at {root}: pass shards= (a count, a list of "
                f"shard numbers in 1..{N_SHARDS} or 'all', 95.9 GB) to download the published "
                "shards, or archive= (folder of downloaded XCM files); see the module docstring")
        else:
            published = download_birdset_xcm(shards, cache_dir)
        used = [int(Path(d).name[10:14]) for d in published[1]]
        source = f"hf:{HF_REPO}@{HF_COMMIT}:shards={used}"
        if len(used) < N_SHARDS:
            warnings.warn(f"BirdSet XCM: {len(used)} of {N_SHARDS} shards; the shards are "
                          "sorted by species, so species are missing.", UserWarning, stacklevel=2)
    params = {"data_dir": source, "features": features, "n_mfcc": n_mfcc,
                  "n_mels": n_mels, "limit": limit, "min_recordings": min_recordings, "crop_s": crop_s,
                  "estimator": estimator}
    run = run_pipeline(
        "birdset_xcm", params=params,
        process_fn=partial(_process, root=root, shards=published, limit=limit,
                           min_recordings=min_recordings, crop_s=crop_s, features=features,
                           n_mfcc=n_mfcc, n_mels=n_mels),
        covariance_fn=partial(estimate_stage, estimator=estimator),
        cache_dir=cache_dir, keep=keep, force=force)
    return load_result(run, birdset_xcm_splits, fold=fold)


@DatasetManager.register_dataset("birdset_xcm")
class BirdSetXCMDataset(SequenceCovarianceDataset):
    """BirdSet XCM covariances (see :mod:`spdnet_datasets.audio.birdset_xcm`).

    Emits a ``UserWarning`` about the partial copy (the fetch warns about missing
    shards when the published shards are used, ``shards=``).
    """

    DOMAIN, NAME = DOMAIN, NAME
    DOWNLOADABLE = True

    def __init__(self, data_dir=None, *, features="mfcc", n_mfcc=20, n_mels=64,
                 min_recordings=10, crop_s=None, shards=None, **kwargs):
        self.feature_kwargs = {"features": features, "n_mfcc": n_mfcc, "n_mels": n_mels,
                                   "min_recordings": min_recordings, "crop_s": crop_s,
                                   "shards": shards}
        if dataset_path(DOMAIN, NAME, data_dir).is_dir():
            warnings.warn(PARTIAL_WARNING, UserWarning, stacklevel=2)
        super().__init__(data_dir, **kwargs)

    def _fetch(self):
        return fetch_birdset_xcm(self.data_dir, fold=self.fold, **self.fetch_kwargs,
                                 **self.feature_kwargs)

    def _make_splits(self, meta, labels, groups):
        return birdset_xcm_splits(meta, labels, groups, fold=self.fold, seed=0)


__all__ = ["BirdSetXCMDataset", "birdset_xcm_splits", "download_birdset_xcm",
           "fetch_birdset_xcm", "ingest_birdset_xcm", "select_shards", "shard_index"]
