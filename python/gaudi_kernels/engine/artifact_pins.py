"""Qualified70-g kernel/bridge bytes. A manifest cannot qualify rebuilt ISA."""
PINS = {'batch.torch': 'fd95be3c7dc08d099489fe1dbe88498a8109fbbc43387c263c85e17cb8f9d159',
 'batch.tpc': 'e9447e75356498df226b7d7daa3c2dac3f20efbab32834fc030000e61bdb68e2',
 'block_fp8.torch': '3c9a092b4461c0990200cb88fb04aab62afe031b4e3b0ce488d0177aac6c6e0a',
 'block_fp8.tpc': ('121173a4dca0c293db042f904cdb16d2d74ab58022a9cb2f76d9a4785b776740',
                   '5d83ddd8f69fca2be4b061ba9b739917957310ebde4e456a034ca897a9c7cc01'),
 'down.torch': '8774a7107f1242a7d5e3626b0166cbac020cb34b47e488c566927f695de08f1d',
 'down.tpc': ('9675942ebd61768cfa16064f9a034ef34f11e42238b0dcdfdca1a5811d73c47a',
              'd9e61596550b15ca37023222cde5d3d2e8712cb85ee0e3dfbdfcde382bad5b97'),
 'folded.torch': '54529adf89e7d1527df03ffbfb71c5047d518f406ed4513c10691108fbc50e78',
 'folded.tpc': 'dd913322ff1d8f2f665afef1b50c28e53f9bedcaca85ca49723434073421614b',
 'native.torch': 'd28693fc5f144ec91e364878ae6b7637012c259e5e45cabc4784985daaab7007',
 'native.tpc': 'b0c3382922a66098974b190b37775f31098517f0d26836a5100e77c944237f71',
 'norm.torch': '329098542f9d1093b4ac01606d85e8511583c62d7fc54aef72ed478555a66968',
 'norm.tpc': ('39729fbc57737381d86b2fe640d1c4ce86b7f66f9b7e55567e5424fc4b17f882',
              'd641a312ef694c178bc40d5e5804fb94ff61d6d4047ac4339e63be3356e606f4'),
 'norm_grid24.torch': ('ebee484a85fd44fb06c9b123e0bd3b63b01bab7c117fd06527af05faf0ef5976',
                       '248b19f9c06db23f62c70a5c4d8272e85a7f17caead9c0ffbc7a758db29bd9a0',
                       '8732afec4808539fbe13fac9fafa49001f1ff45b43edbfb4a629fcb17c80ab23',
                       '0f881ce65b6c8fc40529cc982aa4c84dee8f84d4c54e2c81f0edb6a8880bed8c'),
 'norm_grid24.tpc': ('e8487fcb1f0e58cbab74d5dd49dcc4441db313532dd251fca3499866c1812ebd',
                     '1aaa32187c9cd30a57f6d448867823be9356fc73385b094abc4281728d92e17b',
                     'd0e16def1133c7c36141958056db1fce1342b73b6d43b4d1706a80e575b6b6f1',
                     '517dc7aa8a6b91974e063e776a24c66c68e76dbc09b63c5621c3065669b519b0',
                     '24c3d70da960897d1e5cd7d5c8bfbcbf36f7c791557fe5bc4257545b9b29d828',
                     'f051793cbdecd40bcee76de80a9f404ba69b599f99bbb7cec2dcb41ec7c9446d',
                     '2b4e41e153d38d16484564682d3f28e472b1ad818a4ce815678dd308c579dace'),
 'precision.torch': 'e6990cef2ea29e7d061e0bc1d06949af757b3457ee55150db325bf94cd4f4fae',
 'precision.tpc': 'f3e6f643eff6482218c8c551b112df8f3e50740ca88a7e7200daae005015d1dc',
 'qkv_post.torch': '1556d3a2b97fd6a95323e9a1cdff9e9f9ca62fcae4d8007b6abd129e4c783f2c',
 'qkv_post.tpc': 'f332c7b85a2b8faa3ca55bcc3fff02932dc984b3b7dcaeafadc8653b09f68c8a',
 'reduce_isa.torch': '62e4f3eaae0ad25c5d370bf007ffe9fe7dea7f69bba5b58a18b0c64735d6ecaa',
 'reduce_isa.tpc': 'b60f5fbc597b7314653802d3ecf618d9306349f04b62054242ec8232bee5305a',
 'router_post.torch': '69a59292cf1d8ad106c715686ca79b489d0ce5e2d79ede9b10d77907f284cc20',
 'router_post.tpc': '84edf1dc505c084fbb97b22eb2a0f4050db4bea58787aecea56665ddb192a702',
 'scale_tail.torch': 'c73a48e86e32fb5b531cc97d5c205bc35f28c654820773cd82814702688db410',
 'scale_tail.tpc': ('f45f038808d6ec6ba14ac99445fe7bdce35a523c6f63271a3f37624e57130e07',
                    '02a9f1da5e33933f62a5b4aee75bb02fbc82389553a097a23c74ced0693d6fed'),
 'swa.torch': '75e0fd6509cea4ce3ae85eb8149ac3483fc94e4e69ea99d922365ddd2a767f73',
 'swa.tpc': '0d4453c728f074dd90dcbca4fd4fe737c46be0c546ec058f2e7cc49c70a5a746',
 'swa_av.torch': '984de3e1cf314f98467d51a43ed0e4dfcd602d4663254bdb2d36ae3eb4a6e9db',
 'swa_av.tpc': '24d5a035a7f789bdb30b7b0628d0b735b9044f947566e29ea0d1a439ac45c288',
 'vendor.tpc': '5ba475be35565452f0a458464361dd20bcb4d628eb7fb7d31bc2e287f82fc78c'}

# Optional multirow operator assets. These pins establish byte identity, not
# complete-model/MTP quality. The bundle forwards all six original SWA ELFs.
PINS.update({'swa_bundle.tpc':'7c76d728c79186ddb01ef7718c0482e377465e9e15a6c7bf20378be4c18ca8fc',
             'swa_batch.torch':'c4c4697376933ac75bf66d675bee31794dbee9b581cd42c3a90a89fa53c1a12d'})

# Owned expert backend + unchanged batch provider, one SDK database. Byte
# identity is not a full-model performance or quality claim.
PINS.update({
    'moe_bundle.tpc': (
        '01d687d2b315dea1f223cf9e665c8aea68a6fe91bc820992e24dd5fe1dc20b38',
        '81d825bff9e0370aaec2d965ed2685b5e27110b1af5f27dd80933a98c2314864',
    ),
    'expert.torch': (
        'c4fe2d62d5b177da176ba0794d2ffd4522624721a3ed190121a38887f8e21231',
        '9cb199151c3cb01d15f2377fd8ead7a914714dda89b85f8b50e27693b7ac2658',
        'fecc7bc8f5eff352d91180c29d884c599f1ced9fbe3ee303e6d4c407c5a9e871',
        'b5401d7ac0ccc0af280f10977e26dd7dbdc0153ab8f2d6db7ffc25b478951e45',
        'fcf5f9e095f722d7a9a171673e2befc1032b283b555e2e67ba12e8d7af58733f',
        'ccd1b26b80789c7bceb68d0936f71a09692a96be66c2a1e5305d2027c82d2097',
    ),
    'moe_batch_provider.host': (
        PINS['batch.tpc'],
        '23dfc9f986560be6a13653367f9a2cdae3e6120dc873454fbc7ad6cd4b3e6d94',
    ),
    'moe_expert_provider.host': (
        'b86ddfb38a6a8eda01b986edf4139bcb1a8aea625dceda60871dd395244aef40',
        'b4cb9200a432f30e692cdf79b74d1667662bff9a276e9878a65af2f9da2f6d6f',
        '5f3edb9c7ea471d09b84f327de44e29a84078ba8adc5701a73c2beac21af3a6c',
    ),
})
