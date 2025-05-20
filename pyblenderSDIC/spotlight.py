from __future__ import annotations
import json
from typing import Dict, Tuple, Optional
from py3dframe import Frame, Rotation
import numpy
from numbers import Number

class SpotLight(object):
    r"""
    Represents a spot light in 3D space with a defined position and orientation.

    The spot light orientation is defined by the frame.
    The spot light emits along the z-axis of the frame.

    The frame of the light defines the orientation of the camera in 3D space with (convention OPENCV): 

    - origin: The position of the light in 3D space.
    - x-axis: any direction perpendicular to the optical axis of the light.
    - y-axis: any direction perpendicular to the optical axis of the light.
    - z-axis: The optical axis of the light (from the light to the scene).

    The attribute ``spot_size`` is the apperture angle of the spot light. 
    It is defined as the angle between the two edges of the cone of light emitted by the spot light. 
    The figure below shows the definition of the spot light size in Blender (Image source: Blender Manual)

    .. figure:: ../../../pyblenderSDIC/resources/doc/blender_spot_light_size.png
        :width: 400
        :align: center

        Blender spot light size definition.
    
    The attribute ``spot_blend`` is the proportion of the spot light this a smooth transition between the spot light and the background light.
    Setting the ``spot_blend`` to 0.0 will create a sharp transition between the spot light and the background light.

    Parameters
    ----------
    frame : Frame, optional
        The frame of the camera. (see py3dframe : https://https://artezaru.github.io/py3dframe/), default Frame().

    energy : float, optional
        The energy of the spot light. The energy this light would emit over its entire area if it wasn’t limited by the spot angle, default 1.0.

    spot_size : float, optional
        The size of the spot light in radians, default is numpy.pi.

    spot_blend : float, optional
        The blend value of the spot light between 0 and 1, default 0.0.
    """

    def __init__(
        self,
        frame: Frame = Frame(),
        energy: float = 1.0,
        spot_size: float = numpy.pi,
        spot_blend: float = 0.0,
    ) -> None:
        self.frame = frame
        self.spot_size = spot_size
        self.spot_blend = spot_blend
        self.energy = energy


    # ==============================================
    # Property getters and setters
    # ==============================================
    @property
    def frame(self) -> Frame:
        """
        Get or set the frame of the spot light.

        The spot light emits along the z-axis of the frame.

            The frame of the light defines the orientation of the camera in 3D space with (convention OPENCV): 

            - origin: The position of the light in 3D space.
            - x-axis: any direction perpendicular to the optical axis of the light.
            - y-axis: any direction perpendicular to the optical axis of the light.
            - z-axis: The optical axis of the light (from the light to the scene).

        .. seealso::

            - https://artezaru.github.io/py3dframe/ for more information about the frame.

        Returns
        -------
        Frame
            The frame of the light.
        """
        return self._frame

    @frame.setter
    def frame(self, frame: Frame) -> None:
        if not isinstance(frame, Frame):
            raise ValueError("Frame must be a Frame object.")
        self._frame = frame
    

    @property
    def spot_size(self) -> Optional[float]:
        """
        Get or set the spot_size of the spot light.      

        The attribute ``spot_size`` is the apperture angle of the spot light. 
        It is defined as the angle between the two edges of the cone of light emitted by the spot light. 
        The figure below shows the definition of the spot light size in Blender (Image source: Blender Manual)

        .. figure:: ../../../pyblenderSDIC/resources/doc/blender_spot_light_size.png
            :width: 400
            :align: center

            Blender spot light size definition.  

        Returns
        -------
        float
            The spot size of the light in radians.
        """
        return self._spot_size
    
    @spot_size.setter
    def spot_size(self, spot_size: float) -> None:
        if not isinstance(spot_size, Number):
            raise ValueError("spot_size must be a number.")
        if not 0 <= spot_size <= numpy.pi:
            raise ValueError("spot_size must be between 0 and pi radians (180 degrees).")
        self._spot_size = float(spot_size)



    @property
    def spot_blend(self) -> float:
        """
        Get or set the spot_blend value of the spot light.
        
        The attribute ``spot_blend`` is the proportion of the spot light this a smooth transition between the spot light and the background light.
        Setting the ``spot_blend`` to 0.0 will create a sharp transition between the spot light and the background light.

        Returns
        -------
        float
            The spot blend value of the light between 0 and 1.
        """
        return self._spot_blend

    @spot_blend.setter
    def spot_blend(self, spot_blend: float) -> None:
        if not isinstance(spot_blend, Number):
            raise ValueError("spot_blend must be a number.")
        if not 0 <= spot_blend <= 1.0:
            raise ValueError("spot_blend must be between 0 and 1.")
        self._spot_blend = float(spot_blend)



    @property
    def energy(self) -> float:
        """
        Get or set the energy of the spot light.
        
        The energy of the spot light. The energy this light would emit over its entire area if it wasn’t limited by the spot angle.
        The energy is used to calculate the intensity of the light in the scene.

        Returns
        -------
        float
            The energy of the light.
        """
        return self._energy
    
    @energy.setter
    def energy(self, energy: float) -> None:
        if not isinstance(energy, (int, float)):
            raise ValueError("Energy must be a number.")
        if energy < 0:
            raise ValueError("Energy must be greater than or equal to 0.")
        self._energy = float(energy)


    # =============================================
    # OpenCV and OpenGL methods
    # =============================================
    def get_OpenCV_RT(self) -> Tuple[Rotation, numpy.ndarray]:
        r"""
        Get the rotation and translation of the spotlight in the OpenCV format.

        The axis of the spotlight frame for OpenCV are the same as the spotlight frame.
        Furthermore, the convention for OpenCV is :math:`X_{light} = R X_{world} + T`, convention=4 for py3dframe.

        Returns
        -------
        Rotation
            The rotation of the light.
        
        numpy.ndarray
            The translation of the light with shape (3, 1).
        """
        rotation = self.frame.get_global_rotation(convention=4)
        translation = self.frame.get_global_translation(convention=4)
        return rotation, translation
    
    def set_OpenCV_RT(self, rotation: Rotation, translation: numpy.ndarray) -> None:
        r"""
        Get the rotation and translation of the spotlight in the OpenCV format.

        The axis of the spotlight frame for OpenCV are the same as the spotlight frame.
        Furthermore, the convention for OpenCV is :math:`X_{light} = R X_{world} + T`, convention=4 for py3dframe.

        Parameters
        ----------
        rotation: Rotation
            The rotation of the light.

        translation: numpy.ndarray
            The translation of the light with shape (3, 1).
        """
        self.frame.set_global_rotation(rotation, convention=4)
        self.frame.set_global_translation(translation, convention=4)
    
    def get_OpenGL_RT(self) -> Tuple[Rotation, numpy.ndarray]:
        r"""
        Get the rotation and translation of the light in the OpenGL format.

        The axis of the light frame for OpenGL are different from the SpotLight frame:
        - x-axis: The same as the SpotLight frame : right direction of the light (left to right).
        - y-axis: The opposite of the SpotLight frame : up direction of the light (down to up).
        - z-axis: The opposite of the SpotLight frame : (from the scene to the light).

        Furthermore, the convention for OpenGL is :math:`X_{world} = R X_{cam} + T`, convention=0 for py3dframe.

        .. note::

            The OpenGL format is used in Blender.

        Returns
        -------
        Rotation
            The rotation of the light.
        
        numpy.ndarray
            The translation of the light with shape (3, 1).
        """
        rotation = self.frame.get_global_rotation(convention=0)
        x_axis = rotation.as_matrix()[:, 0].reshape((3, 1))
        y_axis = - rotation.as_matrix()[:, 1].reshape((3, 1))
        z_axis = - rotation.as_matrix()[:, 2].reshape((3, 1))
        rotation = Rotation.from_matrix(numpy.column_stack((x_axis, y_axis, z_axis)))
        translation = self.frame.get_global_translation(convention=0)
        return rotation, translation
    
    def set_OpenGL_RT(self, rotation: Rotation, translation: numpy.ndarray) -> None:
        r"""
        Set the rotation and translation of the light in the OpenGL format.

        The axis of the light frame for OpenGL are different from the SpotLight frame:
        - x-axis: The same as the SpotLight frame : right direction of the light (left to right).
        - y-axis: The opposite of the SpotLight frame : up direction of the light (down to up).
        - z-axis: The opposite of the SpotLight frame : (from the scene to the light).

        Furthermore, the convention for OpenGL is :math:`X_{world} = R X_{cam} + T`, convention=0 for py3dframe.

        .. note::

            The OpenGL format is used in Blender.

        Parameters
        -----------
        rotation: Rotation
            The rotation of the light.
        
        translation: numpy.ndarray
            The translation of the light with shape (3, 1).
        """
        x_axis = rotation.as_matrix()[:, 0].reshape((3, 1))
        y_axis = - rotation.as_matrix()[:, 1].reshape((3, 1))
        z_axis = - rotation.as_matrix()[:, 2].reshape((3, 1))
        rotation = Rotation.from_matrix(numpy.column_stack((x_axis, y_axis, z_axis)))
        self.frame.set_global_rotation(rotation, convention=0)
        self.frame.set_global_translation(translation, convention=0)

    # ==============================================
    # Load and save methods
    # ==============================================
    def save_to_dict(self, description: str = "") -> Dict:
        """
        Export the SpotLight's data to a dictionary.

        The structure of the dictionary is as follows:

        .. code-block:: python

            {
                "type": "SpotLight [pyblenderSDIC]",
                "description": "Description of the spot light",
                "frame": {
                    "translation": [0.0, 0.0, 0.0],
                    "quaternion": [1.0, 0.0, 0.0, 0.0],
                    "convention": 0,
                    "parent": None
                },
                "energy": 10.0,
                "spot_size": 0.985,
                "spot_blend": 0.5
                }
            }

        Parameters
        ----------
        description : str, optional
            A description of the spot light, by default "".

        Returns
        -------
        dict
            A dictionary containing the spot light's data.

        Raises
        ------
        ValueError
            If the description is not a string.
        """
        # Check the description
        if not isinstance(description, str):
            raise ValueError("Description must be a string.")
        
        # Create the dictionary
        data = {
            "type": "SpotLight [pyblenderSDIC]",
            "frame": self.frame.save_to_dict(),
            "energy": self.energy,
            "spot_size": self.spot_size,
            "spot_blend": self.spot_blend
        }

        # Add the description
        if len(description) > 0:
            data["description"] = description
        
        return data



    @classmethod
    def load_from_dict(cls, data: Dict) -> SpotLight:
        """
        Create a SpotLight instance from a dictionary.

        The structure of the dictionary should be as provided by the :meth:`pyblenderSDIC.SpotLight.save_to_dict` method.

        Parameters
        ----------
        data : dict
            A dictionary containing the spot light's data.
        
        Returns
        -------
        SpotLight
            The SpotLight instance.

        Raises
        ------
        ValueError
            If the data is not a dictionary.
        """
        # Check for the input type
        if not isinstance(data, dict):
            raise ValueError("data must be a dictionary.")
        
        # Create the Camera instance
        frame = Frame.load_from_dict(data["frame"])
        spot_size = data.get("spot_size", numpy.pi)
        spot_blend = data.get("spot_blend", 0.0)
        energy = data.get("energy", 1.0)

        return cls(frame=frame, energy=energy, spot_size=spot_size, spot_blend=spot_blend)



    def save_to_json(self, filepath: str, description: str = "") -> None:
        """
        Export the SpotLight's data to a JSON file.

        The structure of the JSON file follows the :meth:`pyblenderSDIC.SpotLight.save_to_dict` method.

        Parameters
        ----------
        filepath : str
            The path to the JSON file.
        
        description : str, optional
            A description of the spot light, by default "".

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
    def load_from_json(cls, filepath: str) -> SpotLight:
        """
        Create a SpotLight instance from a JSON file.

        The structure of the JSON file follows the :meth:`pyblenderSDIC.SpotLight.save_to_dict` method.

        Parameters
        ----------
        filepath : str
            The path to the JSON file.
        
        Returns
        -------
        SpotLight
            A SpotLight instance.
        
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
    

    
    # ==============================================
    # Blender methods
    # ==============================================
    def get_RT_blender(self) -> Tuple[Rotation, numpy.ndarray]:
        """
        Get the rotation and translation of the light in the Blender format.

        The axis of the light frame for Blender are different from the SpotLight frame:
        - x-axis: The same as the SpotLight frame : right direction of the light (left to right).
        - y-axis: The opposite of the SpotLight frame : up direction of the light (down to up).
        - z-axis: The opposite of the SpotLight frame : (from the scene to the light).

        Furthermore, the convention for Blender is :math:`X_{world} = R X_{cam} + T`, convention=0 for py3dframe.

        Returns
        -------
        Rotation
            The rotation of the light.
        
        numpy.ndarray
            The translation of the light with shape (3, 1).
        """
        rotation = self.frame.get_global_rotation(convention=0)
        x_axis = rotation.as_matrix()[:, 0].reshape((3, 1))
        y_axis = - rotation.as_matrix()[:, 1].reshape((3, 1))
        z_axis = - rotation.as_matrix()[:, 2].reshape((3, 1))
        rotation = Rotation.from_matrix(numpy.column_stack((x_axis, y_axis, z_axis)))
        translation = self.frame.get_global_translation(convention=0)
        return rotation, translation