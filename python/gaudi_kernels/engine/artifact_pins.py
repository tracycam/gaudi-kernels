"""Qualified70-g kernel/bridge bytes. A manifest cannot qualify rebuilt ISA."""
PINS = {'batch.torch': 'fd95be3c7dc08d099489fe1dbe88498a8109fbbc43387c263c85e17cb8f9d159',
 'batch.tpc': 'e9447e75356498df226b7d7daa3c2dac3f20efbab32834fc030000e61bdb68e2',
 'block_fp8.torch': '3c9a092b4461c0990200cb88fb04aab62afe031b4e3b0ce488d0177aac6c6e0a',
 'block_fp8.tpc': '121173a4dca0c293db042f904cdb16d2d74ab58022a9cb2f76d9a4785b776740',
 'down.torch': '8774a7107f1242a7d5e3626b0166cbac020cb34b47e488c566927f695de08f1d',
 'down.tpc': '9675942ebd61768cfa16064f9a034ef34f11e42238b0dcdfdca1a5811d73c47a',
 'folded.torch': '54529adf89e7d1527df03ffbfb71c5047d518f406ed4513c10691108fbc50e78',
 'folded.tpc': 'dd913322ff1d8f2f665afef1b50c28e53f9bedcaca85ca49723434073421614b',
 'native.torch': 'd28693fc5f144ec91e364878ae6b7637012c259e5e45cabc4784985daaab7007',
 'native.tpc': 'b0c3382922a66098974b190b37775f31098517f0d26836a5100e77c944237f71',
 'norm.torch': '329098542f9d1093b4ac01606d85e8511583c62d7fc54aef72ed478555a66968',
 'norm.tpc': '39729fbc57737381d86b2fe640d1c4ce86b7f66f9b7e55567e5424fc4b17f882',
 'norm_grid24.torch': 'ebee484a85fd44fb06c9b123e0bd3b63b01bab7c117fd06527af05faf0ef5976',
 'norm_grid24.tpc': 'e8487fcb1f0e58cbab74d5dd49dcc4441db313532dd251fca3499866c1812ebd',
 'precision.torch': 'e6990cef2ea29e7d061e0bc1d06949af757b3457ee55150db325bf94cd4f4fae',
 'precision.tpc': 'f3e6f643eff6482218c8c551b112df8f3e50740ca88a7e7200daae005015d1dc',
 'qkv_post.torch': '1556d3a2b97fd6a95323e9a1cdff9e9f9ca62fcae4d8007b6abd129e4c783f2c',
 'qkv_post.tpc': 'f332c7b85a2b8faa3ca55bcc3fff02932dc984b3b7dcaeafadc8653b09f68c8a',
 'reduce_isa.torch': '62e4f3eaae0ad25c5d370bf007ffe9fe7dea7f69bba5b58a18b0c64735d6ecaa',
 'reduce_isa.tpc': 'b60f5fbc597b7314653802d3ecf618d9306349f04b62054242ec8232bee5305a',
 'router_post.torch': '69a59292cf1d8ad106c715686ca79b489d0ce5e2d79ede9b10d77907f284cc20',
 'router_post.tpc': '84edf1dc505c084fbb97b22eb2a0f4050db4bea58787aecea56665ddb192a702',
 'scale_tail.torch': 'c73a48e86e32fb5b531cc97d5c205bc35f28c654820773cd82814702688db410',
 'scale_tail.tpc': 'f45f038808d6ec6ba14ac99445fe7bdce35a523c6f63271a3f37624e57130e07',
 'swa.torch': '75e0fd6509cea4ce3ae85eb8149ac3483fc94e4e69ea99d922365ddd2a767f73',
 'swa.tpc': '0d4453c728f074dd90dcbca4fd4fe737c46be0c546ec058f2e7cc49c70a5a746',
 'swa_av.torch': '984de3e1cf314f98467d51a43ed0e4dfcd602d4663254bdb2d36ae3eb4a6e9db',
 'swa_av.tpc': '24d5a035a7f789bdb30b7b0628d0b735b9044f947566e29ea0d1a439ac45c288',
 'tensor_hold.host': '1b941c734c3898836289c4b524e742c3c32a9c9355e6337f7906457d76130296',
 'vendor.tpc': '5ba475be35565452f0a458464361dd20bcb4d628eb7fb7d31bc2e287f82fc78c'}

# Optional multirow operator assets. These pins establish byte identity, not
# complete-model/MTP quality. The bundle forwards all six original SWA ELFs.
PINS.update({'swa_bundle.tpc':'7c76d728c79186ddb01ef7718c0482e377465e9e15a6c7bf20378be4c18ca8fc',
             'swa_batch.torch':'c4c4697376933ac75bf66d675bee31794dbee9b581cd42c3a90a89fa53c1a12d'})

# Owned expert backend + unchanged batch provider, one SDK database. Byte
# identity is not a full-model performance or quality claim.
PINS.update({
    'moe_bundle.tpc': '01d687d2b315dea1f223cf9e665c8aea68a6fe91bc820992e24dd5fe1dc20b38',
    'expert.torch': '74cbde5379b3a78d354142173e0a7255a18603eaad94fd12771af4ae822ef8b1',
    'moe_batch_provider.host': PINS['batch.tpc'],
    'moe_expert_provider.host': 'b86ddfb38a6a8eda01b986edf4139bcb1a8aea625dceda60871dd395244aef40',
})
