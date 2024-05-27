import py3Dmol


def draw_mol(file_path):
    with open(file_path, 'r') as fo:
        xyz = fo.read()
    view = py3Dmol.view(width=800, height=400)
    view.addModel(xyz, 'xyz')
    view.setStyle({'sphere': {'scale': 0.35}, 'stick': {'radius': 0.20}})
    view.zoomTo()
    view.show()
