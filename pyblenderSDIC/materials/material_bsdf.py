from __future__ import annotations
import numpy
from typing import Optional, Dict, List, Union, Sequence
from numbers import Number
import json


class MaterialBSDF:
    """
    Represents a material with detailed physical and visual properties for Blender's BSDF Principled shader.
    The class used the Blender's Principled BSDF shader as a base for the material. The parameters are based on the Blender's documentation.

    .. seealso::

        https://docs.blender.org/manual/en/latest/render/shader_nodes/shader/principled.html


    .. note::

        If a pattern will be used for the associated mesh, the base color can be ignored. 
        The pattern will be used as the base color of the material.
        If the pattern and a base color are both set, the pattern will be multiplied by the given base color.

    Parameters
    ----------
    base_color : array_like, optional
        The base color of the material, with RGBA values. Default to None.
    
    metallic : float, optional
        The metallic value of the material. Default to None.
    
    roughness : float, optional
        The roughness value of the material. Default to None.
    
    IOR : float, optional
        The index of refraction of the material. Default to None.

    alpha : float, optional
        The alpha value of the material. Default to None.
    
    normal : array_like, optional
        The normal of the material with XYZ values. Default to None.
    
    weight : float, optional
        The weight of the material. Default to None.

    subsurface_weight : float, optional
        The subsurface weight of the material. Default to None.
    
    subsurface_radius : array_like, optional
        The subsurface radius of the material. Default to None.
    
    subsurface_scale : float, optional
        The subsurface scale of the material. Default to None.
    
    subsurface_IOR : float, optional
        The subsurface index of refraction of the material. Default to None.

    subsurface_anisotropy : float, optional
        The subsurface anisotropy of the material. Default to None.
    
    specular_IOR_level : float, optional
        The specular index of refraction level of the material. Default to None.
    
    specular_tint : array_like, optional
        The specular tint of the material with RGBA values. Default to None.
    
    anisotropic : float, optional
        The anisotropic value of the material. Default to None.

    anisotropic_rotation : float, optional
        The anisotropic rotation of the material. Default to None.
    
    tangent : array_like, optional
        The tangent of the material with XYZ values. Default to None.

    transmission_weight : float, optional
        The transmission weight of the material. Default to None.

    coat_weight : float, optional
        The coat weight of the material. Default to None.
    
    coat_roughness : float, optional
        The coat roughness of the material. Default to None.
    
    coat_IOR : float, optional
        The coat index of refraction of the material. Default to None.
    
    coat_tint : array_like, optional
        The coat tint of the material with RGBA values. Default to None.
    
    coat_normal : array_like, optional
        The coat normal of the material with XYZ values. Default to None.
    
    sheen_weight : float, optional
        The sheen weight of the material. Default to None.
    
    sheen_roughness : float, optional
        The sheen roughness of the material. Default to None.
    
    sheen_tint : array_like, optional
        The sheen tint of the material with RGBA values. Default to None.
    
    emission_color : array_like, optional
        The emission color of the material with RGBA values. Default to None.
    
    emission_strength : float, optional
        The emission strength of the material. Default to None.
    
    thin_film_thickness : float, optional
        The thin film thickness of the material. Default to None.
    
    thin_film_IOR : float, optional
        The thin film index of refraction of the material. Default to None.
    """
    def __init__(
        self,
        base_color: Optional[Union[Sequence[Number], numpy.ndarray]] = None,
        metallic: Optional[Number] = None,
        roughness: Optional[Number] = None,
        IOR: Optional[Number] = None,
        alpha: Optional[Number] = None,
        normal: Optional[Union[Sequence[Number], numpy.ndarray]] = None,
        weight: Optional[Number] = None,
        subsurface_weight: Optional[Number] = None,
        subsurface_radius: Optional[Union[Sequence[Number], numpy.ndarray]] = None,
        subsurface_scale: Optional[Number] = None,
        subsurface_IOR: Optional[Number] = None,
        subsurface_anisotropy: Optional[Number] = None,
        specular_IOR_level: Optional[Number] = None,
        specular_tint: Optional[Union[Sequence[Number], numpy.ndarray]] = None,
        anisotropic: Optional[Number] = None,
        anisotropic_rotation: Optional[Number] = None,
        tangent: Optional[Union[Sequence[Number], numpy.ndarray]] = None,
        transmission_weight: Optional[Number] = None,
        coat_weight: Optional[Number] = None,
        coat_roughness: Optional[Number] = None,
        coat_IOR: Optional[Number] = None,
        coat_tint: Optional[Union[Sequence[Number], numpy.ndarray]] = None,
        coat_normal: Optional[Union[Sequence[Number], numpy.ndarray]] = None,
        sheen_weight: Optional[Number] = None,
        sheen_roughness: Optional[Number] = None,
        sheen_tint: Optional[Union[Sequence[Number], numpy.ndarray]] = None,
        emission_color: Optional[Union[Sequence[Number], numpy.ndarray]] = None,
        emission_strength: Optional[Number] = None,
        thin_film_thickness: Optional[Number] = None,
        thin_film_IOR: Optional[Number] = None,
    ) -> None:
        self.base_color = base_color
        self.metallic = metallic
        self.roughness = roughness
        self.IOR = IOR
        self.alpha = alpha
        self.normal = normal
        self.weight = weight
        self.subsurface_weight = subsurface_weight
        self.subsurface_radius = subsurface_radius
        self.subsurface_scale = subsurface_scale
        self.subsurface_IOR = subsurface_IOR
        self.subsurface_anisotropy = subsurface_anisotropy
        self.specular_IOR_level = specular_IOR_level
        self.specular_tint = specular_tint
        self.anisotropic = anisotropic
        self.anisotropic_rotation = anisotropic_rotation
        self.tangent = tangent
        self.transmission_weight = transmission_weight
        self.coat_weight = coat_weight
        self.coat_roughness = coat_roughness
        self.coat_IOR = coat_IOR
        self.coat_tint = coat_tint
        self.coat_normal = coat_normal
        self.sheen_weight = sheen_weight
        self.sheen_roughness = sheen_roughness
        self.sheen_tint = sheen_tint
        self.emission_color = emission_color
        self.emission_strength = emission_strength
        self.thin_film_thickness = thin_film_thickness
        self.thin_film_IOR = thin_film_IOR

    # Property getters and setters
    @property
    def base_color(self) -> Optional[numpy.ndarray]:
        return self._base_color

    @base_color.setter
    def base_color(self, base_color: Optional[Union[Sequence[Number], numpy.ndarray]]) -> None:
        if base_color is not None:
            base_color = numpy.array(base_color, dtype=numpy.float64).reshape((4,))
        self._base_color = base_color
    


    @property
    def metallic(self) -> Optional[Number]:
        return self._metallic

    @metallic.setter
    def metallic(self, metallic: Optional[Number]) -> None:
        if metallic is not None and not isinstance(metallic, Number):
            raise ValueError("Metallic must be a float.")
        self._metallic = float(metallic) if metallic is not None else None



    @property
    def roughness(self) -> Optional[Number]:
        return self._roughness

    @roughness.setter
    def roughness(self, roughness: Optional[Number]) -> None:
        if roughness is not None and not isinstance(roughness, Number):
            raise ValueError("Roughness must be a float.")
        self._roughness = float(roughness) if roughness is not None else None



    @property
    def IOR(self) -> Optional[Number]:
        return self._IOR

    @IOR.setter
    def IOR(self, IOR: Optional[Number]) -> None:
        if IOR is not None and not isinstance(IOR, Number):
            raise ValueError("IOR must be a float.")
        self._IOR = float(IOR) if IOR is not None else None
    


    @property
    def alpha(self) -> Optional[Number]:
        return self._alpha
    
    @alpha.setter
    def alpha(self, alpha: Optional[Number]) -> None:
        if alpha is not None and not isinstance(alpha, Number):
            raise ValueError("Alpha must be a float.")
        self._alpha = float(alpha) if alpha is not None else None


    @property
    def normal(self) -> Optional[numpy.ndarray]:
        return self._normal

    @normal.setter
    def normal(self, normal: Optional[Union[Sequence[Number], numpy.ndarray]]) -> None:
        if normal is not None:
            normal = numpy.array(normal, dtype=numpy.float64).reshape((3,))
        self._normal = normal



    @property
    def weight(self) -> Optional[Number]:
        return self._weight

    @weight.setter
    def weight(self, weight: Optional[Number]) -> None:
        if weight is not None and not isinstance(weight, Number):
            raise ValueError("Weight must be a float.")
        self._weight = float(weight) if weight is not None else None



    @property
    def subsurface_weight(self) -> Optional[Number]:
        return self._subsurface_weight

    @subsurface_weight.setter
    def subsurface_weight(self, subsurface_weight: Optional[Number]) -> None:
        if subsurface_weight is not None and not isinstance(subsurface_weight, Number):
            raise ValueError("Subsurface weight must be a float.")
        self._subsurface_weight = float(subsurface_weight) if subsurface_weight is not None else None



    @property
    def subsurface_radius(self) -> Optional[numpy.ndarray]:
        return self._subsurface_radius

    @subsurface_radius.setter
    def subsurface_radius(self, subsurface_radius: Optional[Union[Sequence[Number], numpy.ndarray]]) -> None:
        if subsurface_radius is not None:
            subsurface_radius = numpy.array(subsurface_radius, dtype=numpy.float64).reshape((3,))
        self._subsurface_radius = subsurface_radius



    @property
    def subsurface_scale(self) -> Optional[Number]:
        return self._subsurface_scale

    @subsurface_scale.setter
    def subsurface_scale(self, subsurface_scale: Optional[Number]) -> None:
        if subsurface_scale is not None and not isinstance(subsurface_scale, Number):
            raise ValueError("Subsurface scale must be a float.")
        self._subsurface_scale = float(subsurface_scale) if subsurface_scale is not None else None



    @property
    def subsurface_IOR(self) -> Optional[Number]:
        return self._subsurface_IOR

    @subsurface_IOR.setter
    def subsurface_IOR(self, subsurface_IOR: Optional[Number]) -> None:
        if subsurface_IOR is not None and not isinstance(subsurface_IOR, Number):
            raise ValueError("Subsurface IOR must be a float.")
        self._subsurface_IOR = float(subsurface_IOR) if subsurface_IOR is not None else None



    @property
    def subsurface_anisotropy(self) -> Optional[Number]:
        return self._subsurface_anisotropy

    @subsurface_anisotropy.setter
    def subsurface_anisotropy(self, subsurface_anisotropy: Optional[Number]) -> None:
        if subsurface_anisotropy is not None and not isinstance(subsurface_anisotropy, Number):
            raise ValueError("Subsurface anisotropy must be a float.")
        self._subsurface_anisotropy = float(subsurface_anisotropy) if subsurface_anisotropy is not None else None



    @property
    def specular_IOR_level(self) -> Optional[Number]:
        return self._specular_IOR_level

    @specular_IOR_level.setter
    def specular_IOR_level(self, specular_IOR_level: Optional[Number]) -> None:
        if specular_IOR_level is not None and not isinstance(specular_IOR_level, Number):
            raise ValueError("Specular IOR level must be a float.")
        self._specular_IOR_level = float(specular_IOR_level) if specular_IOR_level is not None else None



    @property
    def specular_tint(self) -> Optional[numpy.ndarray]:
        return self._specular_tint

    @specular_tint.setter
    def specular_tint(self, specular_tint: Optional[Union[Sequence[Number], numpy.ndarray]]) -> None:
        if specular_tint is not None:
            specular_tint = numpy.array(specular_tint, dtype=numpy.float64).reshape((4,))
        self._specular_tint = specular_tint



    @property
    def anisotropic(self) -> Optional[Number]:
        return self._anisotropic

    @anisotropic.setter
    def anisotropic(self, anisotropic: Optional[Number]) -> None:
        if anisotropic is not None and not isinstance(anisotropic, Number):
            raise ValueError("Anisotropic must be a float.")
        self._anisotropic = float(anisotropic) if anisotropic is not None else None



    @property
    def anisotropic_rotation(self) -> Optional[Number]:
        return self._anisotropic_rotation

    @anisotropic_rotation.setter
    def anisotropic_rotation(self, anisotropic_rotation: Optional[Number]) -> None:
        if anisotropic_rotation is not None and not isinstance(anisotropic_rotation, Number):
            raise ValueError("Anisotropic rotation must be a float.")
        self._anisotropic_rotation = float(anisotropic_rotation) if anisotropic_rotation is not None else None



    @property
    def tangent(self) -> Optional[numpy.ndarray]:
        return self._tangent

    @tangent.setter
    def tangent(self, tangent: Optional[Union[Sequence[Number], numpy.ndarray]]) -> None:
        if tangent is not None:
            tangent = numpy.array(tangent, dtype=numpy.float64).reshape((3,))
        self._tangent = tangent



    @property
    def transmission_weight(self) -> Optional[Number]:
        return self._transmission_weight

    @transmission_weight.setter
    def transmission_weight(self, transmission_weight: Optional[Number]) -> None:
        if transmission_weight is not None and not isinstance(transmission_weight, Number):
            raise ValueError("Transmission weight must be a float")
        self._transmission_weight = float(transmission_weight) if transmission_weight is not None else None



    @property
    def coat_weight(self) -> Optional[Number]:
        return self._coat_weight

    @coat_weight.setter
    def coat_weight(self, coat_weight: Optional[Number]) -> None:
        if coat_weight is not None and not isinstance(coat_weight, Number):
            raise ValueError("Coat weight must be a float.")
        self._coat_weight = float(coat_weight) if coat_weight is not None else None



    @property
    def coat_roughness(self) -> Optional[Number]:
        return self._coat_roughness

    @coat_roughness.setter
    def coat_roughness(self, coat_roughness: Optional[Number]) -> None:
        if coat_roughness is not None and not isinstance(coat_roughness, Number):
            raise ValueError("Coat roughness must be a float.")
        self._coat_roughness = float(coat_roughness) if coat_roughness is not None else None



    @property
    def coat_IOR(self) -> Optional[Number]:
        return self._coat_IOR

    @coat_IOR.setter
    def coat_IOR(self, coat_IOR: Optional[Number]) -> None:
        if coat_IOR is not None and not isinstance(coat_IOR, Number):
            raise ValueError("Coat IOR must be a float.")
        self._coat_IOR = float(coat_IOR) if coat_IOR is not None else None



    @property
    def coat_tint(self) -> Optional[numpy.ndarray]:
        return self._coat_tint

    @coat_tint.setter
    def coat_tint(self, coat_tint: Optional[Union[Sequence[Number], numpy.ndarray]]) -> None:
        if coat_tint is not None:
            coat_tint = numpy.array(coat_tint, dtype=numpy.float64).reshape((4,))
        self._coat_tint = coat_tint



    @property
    def coat_normal(self) -> Optional[numpy.ndarray]:
        return self._coat_normal

    @coat_normal.setter
    def coat_normal(self, coat_normal: Optional[Union[Sequence[Number], numpy.ndarray]]) -> None:
        if coat_normal is not None:
            coat_normal = numpy.array(coat_normal, dtype=numpy.float64).reshape((3,))
        self._coat_normal = coat_normal
    


    @property
    def sheen_weight(self) -> Optional[Number]:
        return self._sheen_weight

    @sheen_weight.setter
    def sheen_weight(self, sheen_weight: Optional[Number]) -> None:
        if sheen_weight is not None and not isinstance(sheen_weight, Number):
            raise ValueError("Sheen weight must be a float.")
        self._sheen_weight = float(sheen_weight) if sheen_weight is not None else None



    @property
    def sheen_roughness(self) -> Optional[Number]:
        return self._sheen_roughness

    @sheen_roughness.setter
    def sheen_roughness(self, sheen_roughness: Optional[Number]) -> None:
        if sheen_roughness is not None and not isinstance(sheen_roughness, Number):
            raise ValueError("Sheen roughness must be a float.")
        self._sheen_roughness = float(sheen_roughness) if sheen_roughness is not None else None



    @property
    def sheen_tint(self) -> Optional[numpy.ndarray]:
        return self._sheen_tint

    @sheen_tint.setter
    def sheen_tint(self, sheen_tint: Optional[Union[Sequence[Number], numpy.ndarray]]) -> None:
        if sheen_tint is not None:
            sheen_tint = numpy.array(sheen_tint, dtype=numpy.float64).reshape((4,))
        self._sheen_tint = sheen_tint



    @property
    def emission_color(self) -> Optional[numpy.ndarray]:
        return self._emission_color

    @emission_color.setter
    def emission_color(self, emission_color: Optional[Union[Sequence[Number], numpy.ndarray]]) -> None:
        if emission_color is not None:
            emission_color = numpy.array(emission_color, dtype=numpy.float64).reshape((4,))
        self._emission_color = emission_color
    


    @property
    def emission_strength(self) -> Optional[Number]:
        return self._emission_strength

    @emission_strength.setter
    def emission_strength(self, emission_strength: Optional[Number]) -> None:
        if emission_strength is not None and not isinstance(emission_strength, Number):
            raise ValueError("Emission strength must be a float.")
        self._emission_strength = float(emission_strength) if emission_strength is not None else None



    @property
    def thin_film_thickness(self) -> Optional[Number]:
        return self._thin_film_thickness

    @thin_film_thickness.setter
    def thin_film_thickness(self, thin_film_thickness: Optional[Number]) -> None:
        if thin_film_thickness is not None and not isinstance(thin_film_thickness, Number):
            raise ValueError("Thin film thickness must be a float.")
        self._thin_film_thickness = float(thin_film_thickness) if thin_film_thickness is not None else None



    @property
    def thin_film_IOR(self) -> Optional[Number]:
        return self._thin_film_IOR

    @thin_film_IOR.setter
    def thin_film_IOR(self, thin_film_IOR: Optional[Number]) -> None:
        if thin_film_IOR is not None and not isinstance(thin_film_IOR, Number):
            raise ValueError("Thin film IOR must be a float.")
        self._thin_film_IOR = float(thin_film_IOR) if thin_film_IOR is not None else None


    # Overridden methods
    def __repr__(self) -> str:
        """
        Represent the Material object with only non-None parameters.

        Returns:
            str: A string representation of the Material with its non-None attributes.
        """
        non_none_params = {
            key.lstrip('_'): (value.tolist() if isinstance(value, numpy.ndarray) else value)
            for key, value in self.__dict__.items() if value is not None
        }
        return f"MaterialBSDF({', '.join(f'{k}={v}' for k, v in non_none_params.items())})"



    # Save and load methods
    def save_to_dict(self, description: str = "") -> Dict:
        r"""
        Export the MaterialBSDF's data to a dictionary.

        The structure of the dictionary is as follows:

        .. code-block:: python

            {
                "type": "MaterialBSDF [pyblenderSDIC]",
                "description": "Description of the material",
                "base_color": [0.5, 0.5, 0.5, 1.0],
                "metallic": 0.5,
                "roughness": 0.5,
                "IOR": 1.5,
                "...": ...,
            }

        Parameters
        ----------
        description : str, optional
            A description of the material, by default "".

        Returns
        -------
        dict
            A dictionary containing the MaterialBSDF's data.

        Raises
        ------
        ValueError
            If the description is not a string.
        """
        # Check the description
        if not isinstance(description, str):
            raise ValueError("description must be a string.")
        
        # Create the dictionary
        return_dict = {"type": "MaterialBSDF [pyblenderSDIC]",}

        # Add the description if it's not empty
        if description:
            return_dict["description"] = description
        
        # Add the properties to the dictionary
        if self.base_color is not None:
            return_dict["base_color"] = self.base_color.tolist()
        if self.metallic is not None:
            return_dict["metallic"] = self.metallic
        if self.roughness is not None:
            return_dict["roughness"] = self.roughness
        if self.IOR is not None:
            return_dict["IOR"] = self.IOR
        if self.alpha is not None:
            return_dict["alpha"] = self.alpha
        if self.normal is not None:
            return_dict["normal"] = self.normal.tolist()
        if self.weight is not None:
            return_dict["weight"] = self.weight
        if self.subsurface_weight is not None:
            return_dict["subsurface_weight"] = self.subsurface_weight
        if self.subsurface_radius is not None:
            return_dict["subsurface_radius"] = self.subsurface_radius.tolist()
        if self.subsurface_scale is not None:
            return_dict["subsurface_scale"] = self.subsurface_scale
        if self.subsurface_IOR is not None:
            return_dict["subsurface_IOR"] = self.subsurface_IOR
        if self.subsurface_anisotropy is not None:
            return_dict["subsurface_anisotropy"] = self.subsurface_anisotropy   
        if self.specular_IOR_level is not None:
            return_dict["specular_IOR_level"] = self.specular_IOR_level
        if self.specular_tint is not None:
            return_dict["specular_tint"] = self.specular_tint.tolist()
        if self.anisotropic is not None:
            return_dict["anisotropic"] = self.anisotropic
        if self.anisotropic_rotation is not None:
            return_dict["anisotropic_rotation"] = self.anisotropic_rotation
        if self.tangent is not None:
            return_dict["tangent"] = self.tangent.tolist
        if self.transmission_weight is not None:
            return_dict["transmission_weight"] = self.transmission_weight
        if self.coat_weight is not None:
            return_dict["coat_weight"] = self.coat_weight
        if self.coat_roughness is not None:
            return_dict["coat_roughness"] = self.coat_roughness
        if self.coat_IOR is not None:
            return_dict["coat_IOR"] = self.coat_IOR
        if self.coat_tint is not None:
            return_dict["coat_tint"] = self.coat_tint.tolist()
        if self.coat_normal is not None:
            return_dict["coat_normal"] = self.coat_normal.tolist()
        if self.sheen_weight is not None:
            return_dict["sheen_weight"] = self.sheen_weight
        if self.sheen_roughness is not None:
            return_dict["sheen_roughness"] = self.sheen_roughness
        if self.sheen_tint is not None:
            return_dict["sheen_tint"] = self.sheen_tint.tolist()
        if self.emission_color is not None:
            return_dict["emission_color"] = self.emission_color.tolist()
        if self.emission_strength is not None:
            return_dict["emission_strength"] = self.emission_strength
        if self.thin_film_thickness is not None:
            return_dict["thin_film_thickness"] = self.thin_film_thickness
        if self.thin_film_IOR is not None:
            return_dict["thin_film_IOR"] = self.thin_film_IOR

        return return_dict
    

    def save_to_json(self, filepath: str, description: str = "") -> None:
        r"""
        Export the MaterialBSDF's data to a JSON file.

        The structure of the JSON file follows the :meth:`pyblenderSDIC.materials.MaterialBSDF.save_to_dict` method.

        Parameters
        ----------
        filepath : str
            The path to the JSON file.
        
        description : str, optional
            A description of the material, by default "".

        Raises
        ------
        FileNotFoundError
            If the filepath is not a valid path.
        """
        # Create the dictionary
        data = self.save_to_dict(description=description)

        # Save the dictionary to a JSON file
        with open(filepath, "w") as file:
            json.dump(data, file, indent=4)


    
    @classmethod
    def load_from_dict(cls, data: Dict) -> MaterialBSDF:
        r"""
        Create a MaterialBSDF instance from a dictionary.

        The structure of the dictionary should be as provided by the :meth:`pyblenderSDIC.meshes.MaterialBSDF.save_to_dict` method.
        The other fields of the dictionary are ignored.

        .. code-block:: python

            from pyblenderSDIC.materials import MaterialBSDF

            mat_dict = {
                "type": "MaterialBSDF [pyblenderSDIC]",
                "description": "Description of the material",
                "base_color": [0.5, 0.5, 0.5, 1.0],
                "metallic": 0.5,
                "roughness": 0.5,
            }

            # Create a MaterialBSDF instance from the dictionary
            material = MaterialBSDF.load_from_dict(mat_dict)

        .. seealso::

            - :meth:`pyblenderSDIC.meshes.MaterialBSDF.save_to_dict` for saving the mesh to a dictionary.
            - :meth:`pyblenderSDIC.meshes.MaterialBSDF.load_from_json` for loading from a JSON file.

        Parameters
        ----------
        data : dict
            A dictionary containing the material's data.
        
        Returns
        -------
        MaterialBSDF
            The MaterialBSDF instance.

        Raises
        ------
        ValueError
            If the data is not a dictionary.
        KeyError
            If required keys are missing from the dictionary.
        """
        # Check for the input type
        if not isinstance(data, dict):
            raise ValueError("data must be a dictionary.")
        
        # Create a new MaterialBSDF instance
        material = cls()
        
        # Extract the properties from the dictionary
        if "base_color" in data:
            material.base_color = numpy.array(data["base_color"], dtype=numpy.float64).reshape((4,))
        if "metallic" in data:
            material.metallic = float(data["metallic"])
        if "roughness" in data:
            material.roughness = float(data["roughness"])
        if "IOR" in data:
            material.IOR = float(data["IOR"])
        if "alpha" in data:
            material.alpha = float(data["alpha"])
        if "normal" in data:
            material.normal = numpy.array(data["normal"], dtype=numpy.float64).reshape((3,))
        if "weight" in data:
            material.weight = float(data["weight"])
        if "subsurface_weight" in data:
            material.subsurface_weight = float(data["subsurface_weight"])
        if "subsurface_radius" in data:
            material.subsurface_radius = numpy.array(data["subsurface_radius"], dtype=numpy.float64).reshape((3,))
        if "subsurface_scale" in data:
            material.subsurface_scale = float(data["subsurface_scale"])
        if "subsurface_IOR" in data:
            material.subsurface_IOR = float(data["subsurface_IOR"])
        if "subsurface_anisotropy" in data:
            material.subsurface_anisotropy = float(data["subsurface_anisotropy"])
        if "specular_IOR_level" in data:
            material.specular_IOR_level = float(data["specular_IOR_level"])
        if "specular_tint" in data:
            material.specular_tint = numpy.array(data["specular_tint"], dtype=numpy.float64).reshape((4,))
        if "anisotropic" in data:
            material.anisotropic = float(data["anisotropic"])
        if "anisotropic_rotation" in data:
            material.anisotropic_rotation = float(data["anisotropic_rotation"])
        if "tangent" in data:
            material.tangent = numpy.array(data["tangent"], dtype=numpy.float64).reshape((3,))
        if "transmission_weight" in data:
            material.transmission_weight
        if "coat_weight" in data:
            material.coat_weight = float(data["coat_weight"])
        if "coat_roughness" in data:
            material.coat_roughness = float(data["coat_roughness"])
        if "coat_IOR" in data:
            material.coat_IOR = float(data["coat_IOR"])
        if "coat_tint" in data:
            material.coat_tint = numpy.array(data["coat_tint"], dtype=numpy.float64).reshape((4,))
        if "coat_normal" in data:
            material.coat_normal = numpy.array(data["coat_normal"], dtype=numpy.float64).reshape((3,))
        if "sheen_weight" in data:
            material.sheen_weight = float(data["sheen_weight"])
        if "sheen_roughness" in data:
            material.sheen_roughness = float(data["sheen_roughness"])
        if "sheen_tint" in data:
            material.sheen_tint = numpy.array(data["sheen_tint"], dtype=numpy.float64).reshape((4,))
        if "emission_color" in data:
            material.emission_color = numpy.array(data["emission_color"], dtype=numpy.float64).reshape((4,))
        if "emission_strength" in data:
            material.emission_strength = float(data["emission_strength"])
        if "thin_film_thickness" in data:
            material.thin_film_thickness = float(data["thin_film_thickness"])
        if "thin_film_IOR" in data:
            material.thin_film_IOR = float(data["thin_film_IOR"])
            
        return material
    

    @classmethod
    def load_from_json(cls, filepath: str) -> MaterialBSDF:
        r"""
        Create a MaterialBSDF instance from a JSON file.

        The structure of the JSON file follows the :meth:`pyblenderSDIC.materials.MaterialBSDF.save_to_dict` method.

        .. code-block:: python

            from pyblenderSDIC.materials import MaterialBSDF

            # Load the mesh from a JSON file
            material = MaterialBSDF.load_from_json("path/to/mesh.json")

        .. seealso::

            - :meth:`pyblenderSDIC.materials.MaterialBSDF.save_to_json` for saving the material to a JSON file.
            - :meth:`pyblenderSDIC.materials.MaterialBSDF.load_from_dict` for loading from a dictionary.

        Parameters
        ----------
        filepath : str
            The path to the JSON file.
        
        Returns
        -------
        MaterialBSDF
            A MaterialBSDF instance.
        
        Raises
        ------
        FileNotFoundError
            If the filepath is not a valid path.
        """
        # Load the dictionary from the JSON file
        with open(filepath, "r") as file:
            data = json.load(file)
        
        # Create the Frame instance
        return cls.load_from_dict(data)



    # helper method
    @classmethod
    def get_details(cls) -> str:
        """
        Get detailed descriptions of each parameter in the MaterialBSDF class.

        Returns:
            str: A detailed description of all parameters and their roles in Blender materials.
        """
        details = {
            "base_color": "Defines the base color of the material (RGBA). Used for the diffuse component of the material.",
            "metallic": "Controls the metallic appearance of the material. 0 = non-metallic, 1 = fully metallic.",
            "roughness": "Determines how rough the surface is. Low values produce smooth, shiny surfaces; high values produce diffuse, matte surfaces.",
            "IOR": "Index of Refraction. Defines how light bends when entering the material. Important for transparent materials like glass or water.",
            "alpha": "Controls the transparency of the material. 1 = fully opaque, 0 = fully transparent.",
            "normal": "Defines the normal map for the material, used to simulate surface details without modifying geometry.",
            "weight": "General blending weight for combining this material with others.",
            "subsurface_weight": "Controls the intensity of subsurface scattering, simulating light penetration for materials like skin, wax, or marble.",
            "subsurface_radius": "Defines how far light scatters below the surface for each RGB channel.",
            "subsurface_scale": "Scales the depth of subsurface scattering.",
            "subsurface_IOR": "Index of Refraction for the subsurface layer.",
            "subsurface_anisotropy": "Adjusts the directionality of subsurface scattering.",
            "specular_IOR_level": "Controls the level of specular reflection based on the Index of Refraction.",
            "specular_tint": "Applies a color tint to specular reflections, influenced by the base color.",
            "anisotropic": "Simulates anisotropic reflections, such as those found on brushed metal surfaces.",
            "anisotropic_rotation": "Controls the rotation of the anisotropic effect.",
            "tangent": "Defines the tangent direction for anisotropic reflections.",
            "transmission_weight": "Controls the amount of light transmitted through the material. Used for transparent materials.",
            "coat_weight": "Defines the intensity of a clear coat layer applied over the material.",
            "coat_roughness": "Controls the roughness of the clear coat layer.",
            "coat_IOR": "Index of Refraction for the clear coat layer.",
            "coat_tint": "Applies a color tint to the clear coat reflections.",
            "coat_normal": "Defines the normal map specifically for the clear coat layer.",
            "sheen_weight": "Controls the intensity of a velvet-like effect, often used for fabrics.",
            "sheen_roughness": "Adjusts the roughness of the sheen effect.",
            "sheen_tint": "Adds a color tint to the sheen effect.",
            "emission_color": "Defines the color of light emitted by the material (self-illumination).",
            "emission_strength": "Controls the intensity of light emitted by the material.",
            "thin_film_thickness": "Simulates the thickness of a thin film layer, creating iridescent effects.",
            "thin_film_IOR": "Index of Refraction for the thin film layer, affecting the appearance of iridescence.",
        }
        # Format details into a readable string
        return "\n".join(f"{param}: {description}" for param, description in details.items())
