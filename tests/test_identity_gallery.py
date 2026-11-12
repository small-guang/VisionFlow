import unittest
import numpy as np
from core.identity_gallery import IdentityGallery


class IdentityGalleryTests(unittest.TestCase):
    def test_face_enrols_two_people_and_reappearance_reuses_first_identity(self):
        gallery=IdentityGallery(face_threshold=.5,body_threshold=.8)
        a=gallery.observe(face_embedding=[1,0,0],body_embedding=[1,0])
        b=gallery.observe(face_embedding=[0,1,0],body_embedding=[0,1])
        again=gallery.observe(face_embedding=[.9,.1,0],body_embedding=[.99,.1])
        self.assertEqual(a['identity_id'],'Demo-P001')
        self.assertEqual(b['identity_id'],'Demo-P002')
        self.assertEqual(again['identity_id'],'Demo-P001')
        self.assertEqual(again['status'],'face_confirmed')

    def test_body_alone_is_candidate_and_cannot_create_or_pollute_identity(self):
        gallery=IdentityGallery()
        empty=gallery.observe(body_embedding=[1,0])
        self.assertIsNone(empty['identity_id'])
        gallery.observe(face_embedding=[1,0],body_embedding=[1,0])
        candidate=gallery.observe(body_embedding=[1,0])
        self.assertEqual(candidate['candidate_id'],'Demo-P001')
        self.assertEqual(candidate['status'],'body_candidate')
        self.assertIsNone(candidate['identity_id'])
        self.assertEqual(gallery.size,1)

    def test_conflicting_face_takes_precedence_over_similar_body(self):
        gallery=IdentityGallery()
        gallery.observe(face_embedding=[1,0],body_embedding=[1,0])
        result=gallery.observe(face_embedding=[0,1],body_embedding=[1,0])
        self.assertEqual(result['identity_id'],'Demo-P002')
        self.assertTrue(result['body_conflict'])

    def test_invalid_or_changed_embedding_space_is_rejected(self):
        gallery=IdentityGallery()
        for bad in ([0,0],[np.nan,1],[np.inf,1]):
            with self.assertRaises(ValueError): gallery.observe(face_embedding=bad)
        gallery.observe(face_embedding=[1,0])
        with self.assertRaises(ValueError): gallery.observe(face_embedding=[1,0,0])
