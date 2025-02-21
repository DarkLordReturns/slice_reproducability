dataset_paths = {
    'oxpets': 'images_oxpets/oxford-iiit-pet/images',
    'pvoc': 'images_pvoc/VOCdevkit/VOC2012/JPEGImages'
}

explainer_class_mapping = {
    'lime': 'LimeExplainer',
    'GridLime': 'LimeExplainer',
    'slice': 'SliceExplainer'
}

plot_colors = {
    'lime': 'red',
    'GridLime': 'green',
    'slice': 'blue'
}
